#!/usr/bin/env python3
"""Reuse sealed native seven-stage outputs; run only six changed visual inputs."""
from run_rc_h593_m_visual_origin_v1 import *
import run_rc_h593_m_visual_origin_v1 as U
from rc_roma_exact_feature_cache_v2 import clone_tree
import rc_roma_shared_native_cache_v1 as S

CACHE = ROOT/'cache/rc_h593_shared_stages_v1'
EXAUTH = ROOT/'registry/rc_h593_shared_stages_authority_v1_20260923.json'
INSIDE = ROOT/'results/rc_h593_m_inside_v1'
IAUTH = ROOT/'registry/rc_h593_m_inside_authority_v1_20260922.json'
NATIVE_ROWS = {}
NATIVE_META = {}
NATIVE_EXECUTION = {}
REUSE_STATS = collections.Counter()

def prepare():
    assert not EXAUTH.exists()
    ia, va = read(IAUTH), read(AUTH)
    for k in ('profile','workers','operator_authority','stages','queries','candidates'):
        assert ia[k] == va[k], k
    sources = [Path(__file__), Path(U.__file__), ROOT/'programs/rc_roma_shared_native_cache_v1.py',
               ROOT/'programs/rc_roma_exact_feature_cache_v2.py', ROOT/'slurm/rc_h593_shared_stages_v1.sbatch',
               ROOT/'plan/RC_H593_SHARED_STAGES_V1_20260923.md', IAUTH, AUTH]
    write(EXAUTH, dict(status='SHARED_NATIVE_STAGES_AUTHORIZED', sources=[bind(p) for p in sources],
          scope='Read validated inside native stages, preserve six visual interventions and original verifier',
          source_profile=ia['profile'], workers=ia['workers'], full_C128=True, precision_unchanged=True,
          user_instruction='Reuse completed stages before new computation; collect missing stages once for downstream use'))

def load_native(index):
    global NATIVE_META, NATIVE_ROWS, NATIVE_EXECUTION
    a=read(EXAUTH)
    for b in a['sources']: checked(b)
    f=INSIDE/f'query{index:03d}'
    val, iv=read(f/'validation.json'),read(f/'inside_validation.json')
    assert val['status']=='M_VISUAL_ORIGIN_QUERY_PASS' and iv['status']=='M_INSIDE_PATHS_PASS'
    assert val['authority']==iv['authority']==bind(IAUTH)
    checked(iv['payload']);meta=read(checked(val['payload']))
    ia,va=read(IAUTH),read(AUTH)
    for k in ('profile','workers','operator_authority','stages','queries','candidates'):assert ia[k]==va[k]
    assert meta['authority']==bind(IAUTH) and meta['execution_ordinal']==index and meta['label_reads']==0
    rows={}
    for b in meta['parts']:
        part=torch.load(checked(b),map_location='cpu',weights_only=True)
        assert part['authority']==bind(IAUTH) and part['query_id']==meta['query_id']
        for row in part['pairs']:
            pos=row['candidate_position'];assert pos not in rows
            assert row['physical_row']==meta['candidate_physical_rows'][pos]
            assert set(row['arms']['NATIVE']['stages'])==set(V.STAGES)
            rows[pos]=row
    assert sorted(rows)==list(range(128))
    NATIVE_META,NATIVE_ROWS=meta,rows
    NATIVE_EXECUTION=dict(authority=bind(EXAUTH),native_validation=bind(f/'validation.json'),
                          native_payload=val['payload'],native_parts=meta['parts'],source='INSIDE_NATIVE_STAGE_REPLAY')
    return meta,rows

def qualification():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    load_native(0)
    oldval=read(OUT/'query000/validation.json');old=read(checked(oldval['payload']))
    assert oldval['status']=='M_VISUAL_ORIGIN_QUERY_PASS' and oldval['authority']==bind(AUTH)
    for k in ('query_id','query_image','query_geometry','candidate_physical_rows'):assert old[k]==NATIVE_META[k],k
    oldrows={}
    for b in old['parts']:
        for row in torch.load(checked(b),map_location='cpu',weights_only=True)['pairs']:
            pos=row['candidate_position'];oldrows[pos]=row
            assert equal_tree(row['arms']['NATIVE']['stages'],NATIVE_ROWS[pos]['arms']['NATIVE']['stages']),('NATIVE_STAGE_BYTES',pos)
    ws=read(checked(read(AUTH)['workers']))['records'];q,prior,refs=C.load_input(ws[0])
    profile=read(C.PROFILE);C.check_profile(profile);core=C.M.legacy_core(profile);core.cell_means=S.ExactPool()
    model=C.M.gpu_model(profile);qp=Path(q['query_source_path']);assert bind(qp)['sha256']==q['query_source_sha256']
    qgeom,_=C.M.geometry(qp,q['query_source_sha256'],'m-origin-query',q['query_grid_shape'],q['processor_input_frame'],profile['sources']['processor']['sha256'])
    qi=core.oriented(qp);observations=[]
    # Natural first eight candidates, fixed before reading any held labels.
    for pos in range(8):
        ref=refs[q['candidate_physical_rows'][pos]];rp=Path(ref['source_path']);assert bind(rp)['sha256']==ref['source_image_sha256']
        rg,_=C.M.geometry(rp,ref['source_image_sha256'],'m-origin-reference',ref['grid_shape'],'DECODED_RAW_BEFORE_EXIF',profile['sources']['processor']['sha256'])
        ri=core.oriented(rp);outputs={};elapsed={}
        for replay in (False,True):
            ob=V.Observer(model,core,lambda *args:None);torch.cuda.synchronize();start=time.monotonic()
            for arm in V.ARMS:
                ob.select(arm,qgeom,rg,q['query_source_sha256'],ref['source_image_sha256'])
                if replay and arm=='NATIVE':
                    ob.stages=clone_tree(NATIVE_ROWS[pos]['arms']['NATIVE']['stages']);pred=None
                else:pred=model.match(qi,ri)
                stages=ob.stages
                for stage,sides in stages.items():sides['M']=float(torch.sqrt(sides['AB']['weights'].mean()*sides['BA']['weights'].mean()))
                assert equal_tree(stages,oldrows[pos]['arms'][arm]['stages']),('SEALED_STAGE_BYTES',pos,arm,replay)
                if not replay:outputs[arm]=clone_tree(pred)
                elif arm!='NATIVE':assert equal_tree(pred,outputs[arm]),('INTERVENTION_DENSE_BYTES',pos,arm)
            torch.cuda.synchronize();elapsed['replay' if replay else 'fresh']=time.monotonic()-start;ob.close()
        observations.append(dict(candidate=pos,seconds=elapsed,all_seven_stages_bit_exact=True,changed_arm_dense_outputs_bit_exact=True))
        print(dict(event='STAGE_REUSE_PILOT_PAIR',**observations[-1]),flush=True)
    result=dict(status='SHARED_NATIVE_STAGES_FULL128_AND_GPU_PASS',authority=bind(EXAUTH),native_pairs=128,
                stages=7,directions=2,gpu_pairs=8,changed_arms=6,all_native_stage_fields_bit_exact=True,
                changed_arm_dense_outputs_bit_exact=True,labels_read=0,rows=observations,
                scope='Historical all128 native parity plus natural first8 complete changed-arm replay; no general speed guarantee')
    write(CACHE/'qualification.json',result)
    print(result['status'],flush=True)

def worker(a, w, index):
    folder = OUT/f'query{index:03d}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'validation.json').exists():
        checked(read(folder/'validation.json')['payload']); return
    assert index != 0, 'QUERY0_ALREADY_SEALED_USE_PILOT_FOR_QUALIFICATION'
    started = time.monotonic(); q, old, refs = C.load_input(w)
    profile = read(C.PROFILE); C.check_profile(profile)
    core = C.M.legacy_core(profile); model = C.M.gpu_model(profile)
    assert model.threshold is None and model.H_lr == model.W_lr == 800 and model.H_hr == model.W_hr == 1280 and model.bidirectional
    opval = read(ROOT/f'results/rc_h593_quality_operator_v1/query{index:03d}/validation.json')
    operator = read(checked(opval['payload'])); assert operator['query_id'] == w['query_id'] and operator['authority'] == a['operator_authority']
    qp = Path(q['query_source_path']); assert bind(qp)['sha256'] == q['query_source_sha256']
    qgeom, qmeta = C.M.geometry(qp, q['query_source_sha256'], 'm-origin-query', q['query_grid_shape'], q['processor_input_frame'], profile['sources']['processor']['sha256'])
    assert NATIVE_META['query_image']['sha256'] == q['query_source_sha256']
    assert NATIVE_META['query_geometry'] == qmeta and NATIVE_META['candidate_physical_rows'] == q['candidate_physical_rows']
    qi = core.oriented(qp); input_sources = {}; current = {}
    def sink(image_sha, tag, lr, hr):
        key = image_sha+'_'+tag
        if key in input_sources: return
        path = OUT/'inputs'/f'{key}.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists():
                value = dict(authority=bind(AUTH), image_sha256=image_sha, transform=tag,
                    original_image=current[image_sha], low_resolution_shape=list(lr.shape), high_resolution_shape=list(hr.shape),
                    low_resolution_thumbnail64=torch.nn.functional.interpolate(lr, size=(64,64), mode='area').cpu(),
                    high_resolution_thumbnail64=torch.nn.functional.interpolate(hr, size=(64,64), mode='area').cpu())
                save(path,value)
            else:
                value=torch.load(path,map_location='cpu',weights_only=True)
                assert value['authority']==bind(AUTH) and value['image_sha256']==image_sha and value['transform']==tag
        input_sources[key] = bind(path)
    current[q['query_source_sha256']] = dict(path=str(qp),sha256=q['query_source_sha256'])
    observer = V.Observer(model, core, sink)
    torch.cuda.reset_peak_memory_stats(); parts = []
    try:
        for part in range(16):
            path = folder/f'part{part:02d}.pt'
            if path.exists():
                value = torch.load(path, map_location='cpu', weights_only=True)
                assert value['authority'] == bind(AUTH) and value['query_id'] == w['query_id'] and value['part'] == part
                parts.append(bind(path)); continue
            if time.monotonic()-started > a['chunk_seconds']:
                write(folder/f"chunk_{os.environ['SLURM_JOB_ID']}_{part:02d}.json",dict(status='M_ORIGIN_NORMAL_PARTIAL',authority=bind(AUTH),completed_parts=len(parts),index=index))
                print(dict(status='PARTIAL', index=index, parts=len(parts)),flush=True); return
            pairs = []
            for pos in range(part*8,(part+1)*8):
                physical = q['candidate_physical_rows'][pos]; ref = refs[physical]; prior = old['candidates'][pos]
                assert physical == prior['physical_row'] and ref['tokens_sha256'] == prior['reference_tokens_sha256']
                rp = Path(ref['source_path']); assert bind(rp)['sha256'] == ref['source_image_sha256']
                current[ref['source_image_sha256']] = dict(path=str(rp),sha256=ref['source_image_sha256'])
                rg, rmeta = C.M.geometry(rp, ref['source_image_sha256'], 'm-origin-reference', ref['grid_shape'], 'DECODED_RAW_BEFORE_EXIF', profile['sources']['processor']['sha256'])
                ri = core.oriented(rp); arms = {}
                for arm in V.ARMS:
                    observer.select(arm,qgeom,rg,q['query_source_sha256'],ref['source_image_sha256'])
                    if index == pos == 0 and arm == 'NATIVE':
                        observer.close(); plain = model.match(qi,ri)
                        observer = V.Observer(model,core,sink)
                        observer.select(arm,qgeom,rg,q['query_source_sha256'],ref['source_image_sha256'])
                    torch.cuda.synchronize(); t = time.monotonic()
                    if arm == 'NATIVE':
                        cached = NATIVE_ROWS[pos]
                        assert cached['physical_row'] == physical and cached['reference_image']['sha256'] == ref['source_image_sha256']
                        assert cached['reference_geometry'] == rmeta
                        observer.stages = clone_tree(cached['arms']['NATIVE']['stages'])
                        pred = None
                        REUSE_STATS['native_forward_skipped'] += 1
                    else:
                        pred = model.match(qi,ri)
                    torch.cuda.synchronize(); seconds = time.monotonic()-t
                    if index == pos == 0 and arm == 'NATIVE':
                        assert equal_tree(plain,pred), 'INSTRUMENTATION_CHANGED_NATIVE'; del plain
                        write(folder/'instrumentation_parity.json',dict(status='M_ORIGIN_FULL_DENSE_NATIVE_BITS_PASS',authority=bind(AUTH)))
                    stages = observer.stages
                    for stage, sides in stages.items():
                        sides['M'] = float(torch.sqrt(sides['AB']['weights'].mean()*sides['BA']['weights'].mean()))
                    if arm == 'NATIVE':
                        assert C.M.bit_equal(stages['HR1']['AB']['weights'],prior['query_visibility'])
                        assert C.M.bit_equal(stages['HR1']['BA']['weights'],prior['reference_visibility'])
                        assert stages['HR1']['M'].hex() == float(prior['old_scores']['visibility_mass']).hex(), 'NATIVE_M'
                    arms[arm] = dict(stages=stages,seconds=seconds)
                    del pred
                pairs.append(dict(candidate_position=pos,physical_row=physical,
                    reference_image=dict(path=str(rp),sha256=ref['source_image_sha256']),reference_geometry=rmeta,
                    free_content=float(operator['modes']['M0Q0R0']['scores'][pos]['real_score']),arms=arms,
                    inputs={key:source for key,source in input_sources.items() if key.split('_',1)[0] in (q['query_source_sha256'],ref['source_image_sha256'])}))
            save(path,dict(authority=bind(AUTH),query_id=w['query_id'],part=part,pairs=pairs,execution=NATIVE_EXECUTION))
            parts.append(bind(path))
            print(dict(event='M_ORIGIN_PART_SAVED',index=index,pairs=(part+1)*8,seconds=time.monotonic()-started),flush=True)
    finally:
        observer.close()
    # Build one compact score table without copying the stage tensors into JSON.
    scores = {arm:{stage:[] for stage in V.STAGES} for arm in V.ARMS}
    for source in parts:
        value = torch.load(checked(source),map_location='cpu',weights_only=True)
        for pair in value['pairs']:
            input_sources.update(pair['inputs'])
            for arm in V.ARMS:
                for stage in V.STAGES: scores[arm][stage].append(pair['arms'][arm]['stages'][stage]['M'])
    write(folder/'payload.json',dict(status='M_ORIGIN_ALL128_SEALED',authority=bind(AUTH),
        query_id=w['query_id'],execution_ordinal=index,query_image=current[q['query_source_sha256']],query_geometry=qmeta,
        candidate_physical_rows=q['candidate_physical_rows'],raw_winner=operator['winner'],
        parts=parts,masses=scores,operator_source=opval['payload'],input_manifest=input_sources,execution=NATIVE_EXECUTION,
        label_reads=0,model_updates=0))
    write(folder/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started,
        descriptor_cache=dict(observer.stats),gpu_peak_bytes=torch.cuda.max_memory_allocated(),native_reuse=dict(REUSE_STATS)))
    subprocess.run([sys.executable,__file__,'verify','--index',str(index)],check=True)


def execute(index):
    assert read(CACHE/'qualification.json')['status']=='SHARED_NATIVE_STAGES_FULL128_AND_GPU_PASS'
    qv=read(CACHE/'qualification.json');assert qv['authority']==bind(EXAUTH)
    # Preload only sealed label-free source tensors before installing the original visual read guard.
    load_native(index)
    a,ws=guard('worker',index)
    original=C.M.legacy_core
    def factory(profile):
        core=original(profile);core.cell_means=S.ExactPool();return core
    C.M.legacy_core=factory
    worker(a,ws[index],index)
    write(CACHE/'consumers'/f"{os.environ['SLURM_JOB_ID']}_{index}.json",dict(authority=bind(EXAUTH),index=index,
          native_forward_skipped=REUSE_STATS['native_forward_skipped'],native_source=NATIVE_EXECUTION,
          complete=(OUT/f'query{index:03d}/validation.json').exists()))
    # Preserve the existing dispatcher receipt contract, with extra execution provenance.
    pa=ROOT/'registry/rc_h593_shared_pooling_cache_authority_v1_20260922.json'
    write(ROOT/'cache/rc_h593_shared_pooling_v1/consumers/visual'/f"{os.environ['SLURM_JOB_ID']}_{index}.json",
          dict(authority=bind(pa),index=index,complete=(OUT/f'query{index:03d}/validation.json').exists(),execution=bind(EXAUTH)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','qualify','worker','verify'));ap.add_argument('--index',type=int)
    args=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    elif args.stage=='qualify':qualification()
    elif args.stage=='worker':execute(args.index)
    else:
        a,ws=guard('verify',args.index);verify(a,ws[args.index],args.index)
