#!/usr/bin/env python3
"""CPU-only, frozen-property replication on identity groups absent from F71.

No GPU, model fitting or candidate reranking is run. Existing single-image
features are reused; missing descriptors and pair interaction J/P are genuinely
recomputed by the frozen original encoder/matcher on CPU and retained. Eighteen
locked interventions use that same CPU J/P and DPT backend. Only two diagnostic
candidates are evaluated per target-present query; no C128 accuracy is claimed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

RC = Path(__file__).resolve().parents[1]
WS = RC.parents[2]
sys.path[:0] = [str(RC / 'programs'), str(RC / 'src')]
OUT = RC / 'results/rc_m_causal128_attribution_chain_v1/phase_replication'
OLD = RC / 'results/rc_roma_position_factor_v1'
PROFILE = RC / 'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json'
METADATA = RC / 'results/rc_h593_quality_operator_eval_v1/result.json'
WORKERS = RC / 'results/rc_crisp_manual_baseline_v1/H593_workers.json'
F71 = RC / 'results/rc_rebut_qr_qrr_v1/protocol.json'
GALLERY = RC / 'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json'
SELECTED = [78, 82, 93, 96, 100, 101, 104, 108, 118]


class BudgetReached(Exception):
    pass


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).absolute(); h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest()}


def checked(b):
    assert bind(b['path']) == {k: b[k] for k in ('path', 'sha256')}, ('SOURCE_DRIFT', b['path'])
    if 'bytes' in b: assert Path(b['path']).stat().st_size == b['bytes']
    return Path(b['path'])


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if p.exists():
        assert p.read_text() == s, ('IMMUTABLE_JSON_DRIFT', str(p)); return
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp'); tmp.write_text(s)
    try: os.link(tmp, p)
    finally: tmp.unlink(missing_ok=True)


def progress(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); os.replace(tmp, p)


def save(p, value, torch):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), ('IMMUTABLE_TENSOR', str(p))
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as f: torch.save(value, f); f.flush(); os.fsync(f.fileno())
        os.link(tmp, p)
    finally: tmp.unlink(missing_ok=True)


def emit(**value):
    print(json.dumps(value, allow_nan=False), flush=True)


def configure():
    os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    return torch


def environment(torch):
    return dict(torch=str(torch.__version__), python=sys.version, threads=torch.get_num_threads(),
                interop=torch.get_num_interop_threads(), capability=torch.backends.cpu.get_cpu_capability(),
                mkldnn=torch.backends.mkldnn.enabled,
                matcher_amp='CPU bfloat16', head_final='float32', pooling='float32 sigmoid then exact float64 integer-cell mean')


def descriptor(sha):
    for root in (RC/'cache/rc_h593_shared_native_v1/images', OLD/'images'):
        p = root/f'{sha}.json'
        if p.exists():
            d = read(p); b = d['payload']; checked(b)
            if 'image' in d: assert d['image']['sha256'] == sha
            return dict(receipt=bind(p), payload=b, image_sha256=sha)
    catalog_path = RC/'cache/rc_h593_shared_native_v1/catalog.json'
    item = next(x for x in read(catalog_path)['images'] if x['sha256'] == sha)
    checked(item)
    return dict(image_sha256=sha, encoding_needed=True, image=item,
                output_path=str(OUT/'images'/f'{sha}.pt'), image_catalog=bind(catalog_path))


def prepare(torch):
    import rc_roma_property_restore_v2 as transform
    meta = read(METADATA)['rows']; old_ids = set(read(F71)['panel_query_ids'])
    old_groups = {r['component'] for r in meta if r['query_id'] in old_ids}
    candidates = [r for r in meta if r['execution_ordinal'] < 128 and r['component'] not in old_groups]
    selected = {}
    for r in sorted(candidates, key=lambda r: r['execution_ordinal']): selected.setdefault(r['component'], r)
    rows = sorted(selected.values(), key=lambda r: r['execution_ordinal'])
    assert [r['execution_ordinal'] for r in rows] == SELECTED
    workers = {w['execution_ordinal']: w for w in read(WORKERS)['records']}
    identities = {x['physical_row']: x['identity'] for x in read(GALLERY)['records']}
    entries = []
    for r in rows:
        index = r['execution_ordinal']; w = workers[index]
        qp = RC/f'results/rc_h593_quality_operator_v1/query{index:03d}/payload.json'
        q = read(qp); assert q['query_id'] == r['query_id'] == w['query_id']
        inter = torch.load(checked(q['intermediates']), map_location='cpu', weights_only=True, mmap=True)
        assert inter['query_id'] == q['query_id']
        axis = q['candidate_physical_rows']; assert len(axis) == len(set(axis)) == 128
        scores = [float(p['scores']['M0Q0R0']['real_score']) for p in inter['pairs']]
        assert [p['physical_row'] for p in inter['pairs']] == axis
        targets = [i for i, physical in enumerate(axis) if identities[physical] == r['identity']]
        # Selection is sealed before any new M. If duplicate gallery instances
        # share the target identity, use strongest free content, stable index tie.
        target = max(targets, key=lambda i: (scores[i], -i)) if targets else None
        wrongs = [i for i in range(128) if i not in targets]
        wrong = max(wrongs, key=lambda i: (scores[i], -i)) if targets else None
        for b in w['roma'].values(): checked(b)
        rp = torch.load(w['roma']['payload']['path'], map_location='cpu', weights_only=True, mmap=True)
        g = rp['records'][w['source_index']]
        assert g['query_id'] == w['source_query_id'] and g['candidate_physical_rows'] == axis
        assert g['query_source_sha256'] == w['source_image_sha256']
        im_path = RC/f'results/rc_h593_m_inside_v1/query{index:03d}/inside_manifest.json'
        im = read(im_path); assert im['query_id'] == w['query_id']
        pairs = []
        for pos in sorted([target, wrong]) if targets else []:
            candidate = g['candidates'][pos]; assert candidate['physical_row'] == axis[pos]
            native = im['records'][pos]; assert native['physical_row'] == axis[pos]
            pairs.append(dict(position=pos, physical_row=axis[pos], role='target' if pos == target else 'fixed_wrong',
                descriptors=[descriptor(g['query_source_sha256']), descriptor(candidate['reference_image_sha256'])],
                image_shas=[g['query_source_sha256'], candidate['reference_image_sha256']],
                geometry_metadata=[g['query_geometry'], candidate['reference_geometry']],
                prior_gpu=im['sources'][pos], prior_gpu_coarse_M=float(native['branches']['A1J1P1']['COARSE'])))
        entries.append(dict(index=index, query_id=w['query_id'], group=r['component'], fold=r['fold'],
            axis=axis, target_positions=targets, target_position=target, fixed_wrong_position=wrong,
            target_present=bool(targets), pairs=pairs, source_worker=w,
            sources=dict(operator_payload=bind(qp), free_content=q['intermediates'], inside_manifest=bind(im_path)),
            free_content_scores=scores))
        del inter, rp
    romadir = WS/'third_party/RoMaV2/src/romav2'
    sources = [Path(__file__), RC/'programs/probe_rc_roma_property_restore_cpu_v3.py',
               RC/'programs/rc_roma_property_restore_v2.py', RC/'programs/rc_roma_property_restore_v1.py',
               RC/'programs/rc_roma_position_factor_v1.py', RC/'programs/rc_roma_shared_native_cache_v1.py',
               RC/'src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py']
    sources += sorted(romadir.rglob('*.py'))
    dinoroot = Path(read(PROFILE)['source_trees']['dinov3']['root'])
    sources += sorted((dinoroot/'dinov3').rglob('*.py'))
    sources += [dinoroot/'hubconf.py', RC/'programs/run_romav2_colnomic_visibility_xf_six_case_v1.py']
    missing = {d['image_sha256']:d for w in entries for pair in w['pairs'] for d in pair['descriptors'] if d.get('encoding_needed')}
    pilot_old = torch.load(checked(read(OLD/'pilot_validation.json')['pair']),weights_only=True,mmap=True,map_location='cpu')
    pilot_cap = torch.load(checked(pilot_old['capture']),weights_only=True,mmap=True,map_location='cpu')
    pilot_image = next(x for x in read(RC/'cache/rc_h593_shared_native_v1/catalog.json')['images'] if x['sha256']==pilot_cap['query_sha'])
    p = dict(status='PHASE_NEWGROUP_PROTOCOL_FROZEN', version=1, code_sources=[bind(x) for x in sources],
        selection_sources=dict(metadata=bind(METADATA), F71=bind(F71), workers=bind(WORKERS), gallery=bind(GALLERY)),
        profile=bind(PROFILE), checkpoint=read(PROFILE)['sources']['checkpoint'],
        head_state=read(OLD/'pilot_validation.json')['head_state'],
        old_manifest=bind(OLD/'manifest.json'), old_pilot_pair=read(OLD/'pilot_validation.json')['pair'],
        old_development_groups=sorted(old_groups), eligible_indices=[r['execution_ordinal'] for r in candidates],
        workers=entries, indices=SELECTED, new_groups=len(entries), arms=list(transform.ARMS),
        phase_arms=list(transform.PHASE_ARMS), seed=transform.SEED, delta=transform.DELTA,
        radius=read(OLD/'manifest.json')['radius'], threads=4, environment=environment(torch),
        scope='FIXED_TWO_DIAGNOSTIC_CANDIDATES_PER_TARGET_PRESENT_QUERY',
        selection='Among first128 groups absent from F71, minimum ordinal in every group; metadata only, no M/outcome selection.',
        fixed_pair='Strongest free-content target and strongest free-content wrong, index tie-break; locked before new M.',
        endpoint='New CPU pair J/P and same-CPU eighteen DPT arms; frozen weights, retained single-image features.',
        primary_contrast='phase_restoration_after_amplitude_permutation',
        secondary_contrasts=['amplitude_restoration_under_local_phase','factorial_interaction','phase_restoration_native_amplitude'],
        new_gpu_forwards=0, new_image_encoder_forwards_required=len(missing), missing_images=list(missing.values()),
        pilot_encoder_image=pilot_image, encoder='Frozen DINOv3 ViT-L/16 original 11/17 layers, CPU BF16, local pretrained f.* weights',
        new_cpu_matcher_forwards_required=True, all_science_cells_same_cpu_backend=True,
        untouched_confirmation=False,
        boundary='Groups disjoint from F71 mechanism development, already used in historical H593 system evaluation. Computational restoration, not true-geometry recovery, unique causality, C128 ranking, or accuracy.')
    checked(p['checkpoint']); checked(p['head_state'])
    write(OUT/'protocol.json', p)
    write(OUT/'preparation_validation.json', dict(status='PHASE_NEWGROUP_PREPARATION_PASS', protocol=bind(OUT/'protocol.json'),
        new_groups=len(entries), queries=len(entries), pairs=sum(len(w['pairs']) for w in entries),missing_single_image_features=len(missing),
        disjoint_from_F71=True, no_new_m_values_read=True, new_gpu_jobs=0))
    emit(status='PREPARED', indices=SELECTED, pairs=sum(len(w['pairs']) for w in entries),missing_single_image_features=len(missing))
    return 0


def guard(torch, gate=True):
    p = read(OUT/'protocol.json')
    for b in p['code_sources']: checked(b)
    assert p['environment'] == environment(torch), ('CPU_RUNTIME_DRIFT', environment(torch), p['environment'])
    assert p['indices'] == SELECTED and p['threads'] == 4
    if gate:
        g = read(OUT/'pilot_validation.json')
        assert g['status'] == 'PHASE_NEWGROUP_CPU_PILOT_PASS' and g['protocol'] == bind(OUT/'protocol.json')
    return p


def model(p, torch):
    # Import the original modules with a CPU device, without package __init__
    # loading image encoders or downloading anything.
    import probe_rc_roma_property_restore_cpu_v3 as h
    h.load_cpu_dpt(torch)
    module = importlib.import_module('romav2.matcher')
    matcher = module.Matcher(module.Matcher.Cfg()).cpu().eval()
    state = torch.load(checked(p['checkpoint']), map_location='cpu', weights_only=True, mmap=True)
    weights = {k[len('matcher.'):]: v for k, v in state.items() if k.startswith('matcher.')}
    matcher.load_state_dict(weights, strict=True)
    old = torch.load(checked(p['head_state']), map_location='cpu', weights_only=True, mmap=True)
    assert all(torch.equal(value, old['head'][key]) for key, value in matcher.head.state_dict().items())
    assert torch.equal(matcher.omega, old['omega']) and torch.equal(matcher.scale, old['scale'])
    for param in matcher.parameters(): param.requires_grad_(False)
    return matcher, module, h


def encoder(p, torch):
    feature_module = importlib.import_module('romav2.features')
    dino = read(checked(p['profile']))['source_trees']['dinov3']['root']
    original_load = torch.hub.load
    def local_load(*args, **kwargs):
        assert kwargs['model']=='dinov3_vitl16' and not kwargs['pretrained']
        return original_load(dino,'dinov3_vitl16',source='local',pretrained=False)
    torch.hub.load = local_load
    try: net=feature_module.Descriptor(feature_module.Descriptor.Cfg()).cpu().eval()
    finally: torch.hub.load=original_load
    state=torch.load(checked(p['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
    net.load_state_dict({k[2:]:v for k,v in state.items() if k.startswith('f.')},strict=True)
    return net


def encode_image(net, item, torch):
    import numpy as np
    from PIL import Image
    checked(item);tick=time.monotonic()
    with Image.open(item['path']) as im:
        try: orientation=int(im.getexif().get(274,1))
        except (TypeError,ValueError): orientation=1
        image=im.copy()
    operation={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,
        5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}.get(orientation)
    if operation is not None:image=image.transpose(operation)
    arr=np.array(image.convert('RGB'));image.close()
    raw=torch.from_numpy(arr).permute(2,0,1).float().div(255)[None]
    lr=torch.nn.functional.interpolate(raw,size=(800,800),mode='bicubic',align_corners=False,antialias=True)
    with torch.inference_mode(): f=[t.clone() for t in net(lr)]
    assert len(f)==2 and all(tuple(t.shape)==(1,50,50,1024) and bool(torch.isfinite(t).all()) for t in f)
    return dict(features=f,seconds=time.monotonic()-tick,image=item,profile=bind(PROFILE),CPU_encoder=True)


def features(pair, torch, p, encoder_box, deadline):
    vals = []
    for d in pair['descriptors']:
        if d.get('encoding_needed'):
            path=Path(d['output_path']);path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not path.exists():
                    pilot_times=read(OUT/'pilot_validation.json')['encoder_seconds']
                    if deadline-time.monotonic()<max(30,1.25*max(pilot_times)+15): raise BudgetReached()
                    if not encoder_box:encoder_box.append(encoder(p,torch))
                    x=encode_image(encoder_box[0],d['image'],torch);x['protocol']=bind(OUT/'protocol.json')
                    save(path,x,torch);emit(stage='MISSING_IMAGE_CPU_ENCODED',sha=d['image_sha256'],seconds=x['seconds'])
                x=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                assert x['protocol']==bind(OUT/'protocol.json') and x['image']==d['image']
        else:
            checked(d['receipt'])
            x = torch.load(checked(d['payload']), map_location='cpu', weights_only=True, mmap=True)
        assert x.get('profile', bind(PROFILE)) == bind(PROFILE)
        f = [t.clone() for t in x['features']]
        assert len(f) == 2 and all(tuple(t.shape) == (1, 50, 50, 1024) and t.dtype in (torch.float32,torch.bfloat16) for t in f)
        vals.append(f)
    return vals


def geometries(pair):
    from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import build_colnomic_canonical_geometry_v2
    result = []
    for side, (meta, sha) in enumerate(zip(pair['geometry_metadata'], pair['image_shas'])):
        g = build_colnomic_canonical_geometry_v2(source_image_sha256=sha, source_key=f'newgroup:{sha}:{side}',
            processor_config_sha256=meta['processor_config_sha256'], raw_size_hw=tuple(meta['raw_size_hw']),
            exif_orientation=int(meta['exif_orientation']), merged_grid_shape=tuple(meta['grid_shape']),
            processor_input_frame=meta['processor_input_frame'])
        assert tuple(g.oriented_size_hw) == tuple(meta['oriented_size_hw'])
        result.append(g)
    return result


def capture(matcher, module, fs, torch):
    sides = []; original = module._compute_head_preds; started = time.monotonic()
    def observe(**kwargs):
        rec = dict(J=kwargs['f_mv_A'].clone(), P=kwargs['match_emb_AB'].clone())
        warp, confidence = original(**kwargs)
        rec.update(warp=warp.clone(), confidence=confidence.clone()); sides.append(rec)
        return warp, confidence
    module._compute_head_preds = observe
    try:
        with torch.inference_mode():
            matcher([t.clone() for t in fs[0]], [t.clone() for t in fs[1]], img_A=None, img_B=None, bidirectional=True)
    finally: module._compute_head_preds = original
    assert len(sides) == 2
    assert all(bool(torch.isfinite(t).all()) for s in sides for t in s.values())
    return dict(sides=sides, seconds=time.monotonic()-started)


def pilot(torch, budget=430):
    p = guard(torch, gate=False); pb = bind(OUT/'protocol.json')
    if (OUT/'pilot_validation.json').exists():
        v = read(OUT/'pilot_validation.json'); assert v['protocol'] == pb; return 0
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'pilot.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started=time.monotonic();matcher, module, h = model(p, torch)
        old_pair = torch.load(checked(p['old_pilot_pair']), weights_only=True, mmap=True, map_location='cpu')
        old_cap = torch.load(checked(old_pair['capture']), weights_only=True, mmap=True, map_location='cpu')
        fs = [[t.clone() for t in torch.load(checked(b), weights_only=True, mmap=True, map_location='cpu')['features']]
              for b in old_cap['descriptor_sources']]
        net=None;encoded=[]
        for i in range(2):
            path=OUT/f'pilot_encoder_{i}.pt'
            if not path.exists():
                if time.monotonic()-started>budget:return 75
                if net is None:net=encoder(p,torch)
                rec=encode_image(net,p['pilot_encoder_image'],torch);rec['protocol']=pb;save(path,rec,torch)
                emit(stage='PILOT_ENCODER_SAVED',repeat=i,seconds=rec['seconds'])
            rec=torch.load(path,weights_only=True,mmap=True,map_location='cpu');assert rec['protocol']==pb;encoded.append(rec)
        encoder_repeat=[h.differences(a,b,torch) for a,b in zip(encoded[0]['features'],encoded[1]['features'])]
        assert all(x['bit_exact'] for x in encoder_repeat),'CPU_ENCODER_REPEAT_NOT_EXACT'
        encoder_gpu_drift=[h.differences(a,b,torch) for a,b in zip(encoded[0]['features'],fs[0])]
        del net
        captures=[]
        for i in range(2):
            path=OUT/f'pilot_matcher_{i}.pt'
            if not path.exists():
                if time.monotonic()-started>budget:return 75
                rec=capture(matcher,module,fs,torch);rec['protocol']=pb;save(path,rec,torch)
            rec=torch.load(path,weights_only=True,mmap=True,map_location='cpu');assert rec['protocol']==pb;captures.append(rec)
        first,second=captures
        if max(x['seconds'] for x in encoded+captures)>320:
            write(OUT/'pilot_cost_blocked.json',dict(status='CPU_PILOT_ATOMIC_FORWARD_EXCEEDS_SHORT_JOB_BUDGET',protocol=pb,
                encoder_seconds=[x['seconds'] for x in encoded],matcher_seconds=[x['seconds'] for x in captures],
                atomic_forward_limit_seconds=320,scientific_negative_result=False))
            return 2
        repeats = []; drift = []
        for side in (0, 1):
            checks = {k: h.differences(first['sides'][side][k], second['sides'][side][k], torch)
                      for k in ('J','P','warp','confidence')}
            assert all(v['bit_exact'] for v in checks.values()), 'CPU_MATCHER_NATIVE_REPEAT_NOT_EXACT'
            repeats.append(checks)
            gpu = old_cap['sides'][side]
            row = {k: h.differences(first['sides'][side][k], gpu[k]['tensor'].to(dtype=getattr(torch, gpu[k]['original_dtype'])), torch)
                   for k in ('J','P')}
            row['confidence'] = h.differences(first['sides'][side]['confidence'], old_pair['branches']['NATIVE'][side]['confidence'], torch)
            drift.append(row)
        import rc_roma_property_restore_v2 as transform
        algebra = transform.self_test()
        save(OUT/'pilot_cpu_capture.pt', dict(protocol=pb, old_gpu_pair=p['old_pilot_pair'], **first), torch)
        write(OUT/'pilot_validation.json', dict(status='PHASE_NEWGROUP_CPU_PILOT_PASS', protocol=pb,
            pilot_source='Historical opened query000 pair000; no new replication group outcome used',
            capture=bind(OUT/'pilot_cpu_capture.pt'), repeat_checks=repeats,
            gpu_drift=drift, gpu_replay_bit_exact_claimed=False, gpu_tolerance_gate=False,
            acceptance='Same-CPU native full matcher repeat exact, finite outputs, checkpoint/head identity, original algebra',
            algebra=algebra, matcher_config=asdict(matcher.cfg), seconds=[first['seconds'], second['seconds']],
            encoder_repeat=encoder_repeat,encoder_gpu_drift=encoder_gpu_drift,
            encoder_seconds=[x['seconds'] for x in encoded],new_single_image_features_required=p['new_image_encoder_forwards_required'],
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        emit(status='PHASE_NEWGROUP_CPU_PILOT_PASS', seconds=[first['seconds'],second['seconds']])
    return 0


def worker(index, budget, torch):
    import rc_roma_property_restore_v2 as transform
    from rc_roma_shared_native_cache_v1 import ExactPool
    p = guard(torch); pb = bind(OUT/'protocol.json'); w = next(w for w in p['workers'] if w['index'] == index)
    folder = OUT/f'query{index:03d}'; folder.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder/'validation.json').exists():
            v = read(folder/'validation.json'); assert v['protocol'] == pb
            for b in v['pairs']: checked(b)
            return 0
        if not w['pairs']:
            write(folder/'validation.json', dict(status='PHASE_NEWGROUP_QUERY_PASS', protocol=pb, index=index,
                query_id=w['query_id'], group=w['group'], target_present=False, pairs=[])); return 0
        matcher, module, h = model(p, torch); pool = ExactPool(); completed = []; fresh = 0;encoder_box=[]
        for pair in w['pairs']:
            pos = pair['position']; d = folder/f'pair{pos:03d}'; d.mkdir(exist_ok=True)
            if (d/'validation.json').exists():
                v = read(d/'validation.json'); assert v['protocol'] == pb
                for a in v['arms']: checked(a['payload'])
                completed.append(bind(d/'validation.json')); continue
            if time.monotonic()-start > budget:
                progress(folder/'progress.json', dict(index=index, completed_pairs=len(completed), new_arms=fresh)); return 75
            try: fs = features(pair, torch, p, encoder_box, start+budget)
            except BudgetReached:
                progress(folder/'progress.json',dict(index=index,stage='ENCODER_BUDGET_CHECKPOINT',new_arms=fresh));return 75
            gs = geometries(pair)
            cap_path = d/'capture.pt'
            if cap_path.exists():
                cap = torch.load(cap_path, weights_only=True, mmap=True, map_location='cpu'); assert cap['protocol'] == pb and cap['pair'] == pair
            else:
                if start+budget-time.monotonic()<1.25*max(read(OUT/'pilot_validation.json')['seconds'])+15:
                    progress(folder/'progress.json',dict(index=index,stage='MATCHER_BUDGET_CHECKPOINT',new_arms=fresh));return 75
                cap = dict(protocol=pb, pair=pair, **capture(matcher,module,fs,torch)); save(cap_path,cap,torch)
                emit(stage='NEW_CPU_PAIR_CAPTURED',index=index,position=pos,seconds=cap['seconds'])
            records = []
            with torch.inference_mode():
                for arm in p['arms']:
                    path = d/f'{arm}.pt'
                    if path.exists():
                        row = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
                        assert row['protocol'] == pb and row['capture'] == bind(cap_path) and row['arm'] == arm
                        records.append(dict(arm=arm,payload=bind(path),M=row['M'])); continue
                    if time.monotonic()-start > budget:
                        progress(folder/'progress.json', dict(index=index, completed_pairs=len(completed), new_arms=fresh)); return 75
                    sides = []; tick = time.monotonic()
                    for side in (0,1):
                        s = cap['sides'][side]
                        changed, invariants = transform.transform(s['P'],matcher.omega,matcher.scale,arm,p['radius'][side],side)
                        values = [t.clone() for t in fs[side]]; values[-1] = values[-1] + s['J'] + changed
                        pred = matcher.head(values,img_A=None,img_B=None)
                        confidence = pred[...,2:]; assert bool(torch.isfinite(confidence).all())
                        if arm == 'NATIVE':
                            assert torch.equal(confidence,s['confidence']), 'NATIVE_CPU_CAPTURE_DPT_REPLAY_DRIFT'
                        weights = pool(confidence[0,...,0].sigmoid(),gs[side])
                        assert weights.dtype == torch.float64 and bool(torch.isfinite(weights).all())
                        sides.append(dict(weights=weights,confidence=confidence.clone(),invariants=invariants))
                    mass = float(torch.sqrt(sides[0]['weights'].mean()*sides[1]['weights'].mean()))
                    assert math.isfinite(mass) and mass > 0
                    row = dict(protocol=pb,index=index,query_id=w['query_id'],position=pos,physical_row=pair['physical_row'],
                        arm=arm,M=mass,sides=sides,capture=bind(cap_path),seconds=time.monotonic()-tick)
                    if arm == 'NATIVE':
                        prior = torch.load(checked(pair['prior_gpu']),weights_only=True,mmap=True,map_location='cpu')
                        row['gpu_drift'] = dict(M=pair['prior_gpu_coarse_M'], logM_difference=math.log(mass)-math.log(pair['prior_gpu_coarse_M']),
                            weights=[h.differences(sides[s]['weights'],prior['sides'][s]['coarse']['A1J1P1']['weights'],torch) for s in (0,1)],
                            diagnostic_only=True)
                    save(path,row,torch); records.append(dict(arm=arm,payload=bind(path),M=mass));fresh += 1
                    emit(stage='ARM_SAVED',index=index,position=pos,arm=arm,seconds=row['seconds'])
            write(d/'validation.json',dict(status='PHASE_NEWGROUP_PAIR_PASS',protocol=pb,index=index,position=pos,
                capture=bind(cap_path),role=pair['role'],arms=records))
            completed.append(bind(d/'validation.json'))
        write(folder/'validation.json',dict(status='PHASE_NEWGROUP_QUERY_PASS',protocol=pb,index=index,query_id=w['query_id'],
            group=w['group'],target_present=w['target_present'],pairs=completed,arms=p['arms']))
        emit(status='PHASE_NEWGROUP_QUERY_PASS',index=index)
    return 0


def contrasts(values):
    def avg(prefix, scope):
        return sum(values[prefix+f'{scope}_{axis}_{sign}'] for axis in ('X','Y') for sign in ('PLUS','MINUS'))/4
    a0g,a0l,a1g,a1l=(avg('', 'GLOBAL'),avg('', 'LOCAL'),avg('AMP_PERMUTE__','GLOBAL'),avg('AMP_PERMUTE__','LOCAL'))
    return dict(phase_restoration_after_amplitude_permutation=a1g-a1l,
        amplitude_restoration_under_local_phase=a0l-a1l,
        factorial_interaction=(a0g-a0l)-(a1g-a1l), phase_restoration_native_amplitude=a0g-a0l)


def join(torch):
    import numpy as np
    p = guard(torch); pb=bind(OUT/'protocol.json'); rows=[];sources=[]; independent=[]
    for w in p['workers']:
        f=OUT/f"query{w['index']:03d}/validation.json"
        if not f.exists(): emit(status='WAITING_FOR_QUERY',index=w['index']);return 75
        v=read(f);assert v['status']=='PHASE_NEWGROUP_QUERY_PASS' and v['protocol']==pb
        assert v['index']==w['index'] and v['query_id']==w['query_id'] and v['group']==w['group']
        assert len(v['pairs'])==len(w['pairs'])
        sources.append(bind(f)); masses={}
        for b in v['pairs']:
            pair=read(checked(b));assert pair['protocol']==pb
            expected=next(x for x in w['pairs'] if x['position']==pair['position'])
            assert pair['index']==w['index'] and pair['role']==expected['role']
            checked(pair['capture']); cells={}
            assert [a['arm'] for a in pair['arms']]==p['arms']
            for a in pair['arms']:
                row=torch.load(checked(a['payload']),weights_only=True,mmap=True,map_location='cpu')
                assert row['protocol']==pb and row['arm']==a['arm'] and row['capture']==pair['capture']
                assert row['index']==w['index'] and row['position']==expected['position'] and row['physical_row']==expected['physical_row']
                u,rv=[side['weights'].double().numpy() for side in row['sides']]
                assert all(np.isfinite(x).all() and ((x>=0)&(x<=1)).all() for x in (u,rv))
                m=math.sqrt(math.fsum(u.tolist())/len(u)*math.fsum(rv.tolist())/len(rv))
                assert abs(m-row['M'])<2e-12 and math.isfinite(m) and m>0
                cells[a['arm']]=m
                independent.append(dict(index=w['index'],position=pair['position'],arm=a['arm'],M=m,payload=a['payload']))
            masses[pair['role']]=cells
        if w['target_present']:
            assert set(masses)=={'target','fixed_wrong'}
            measures={'target_logM':{a:math.log(masses['target'][a]) for a in p['arms']},
                      'fixed_wrong_logM':{a:math.log(masses['fixed_wrong'][a]) for a in p['arms']}}
            measures['fixed_target_wrong_logM_gap']={a:measures['target_logM'][a]-measures['fixed_wrong_logM'][a] for a in p['arms']}
            rows.append(dict(index=w['index'],query_id=w['query_id'],group=w['group'],masses=masses,
                             contrasts={k:contrasts(vals) for k,vals in measures.items()}))
    summary={}
    for metric in ('target_logM','fixed_wrong_logM','fixed_target_wrong_logM_gap'):
        summary[metric]={}
        for contrast in [p['primary_contrast'],*p['secondary_contrasts']]:
            x=np.asarray([r['contrasts'][metric][contrast] for r in rows]);rng=np.random.default_rng(20260927)
            boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1) if len(x) else np.asarray([np.nan])
            summary[metric][contrast]=dict(groups=len(x),group_equal_mean=float(x.mean()),positive_groups=int((x>0).sum()),
                negative_groups=int((x<0).sum()),exploratory_bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))))
    result=dict(status='PHASE_NEWGROUP_REPLICATION_COMPLETE',protocol=pb,rows=rows,summary=summary,
        selected_queries=len(p['workers']),target_present_groups=len(rows),candidate_arms=len(independent),
        population_boundary=p['boundary'],accuracy_claim=False,unique_cause_claim=False)
    write(OUT/'result.json',result);write(OUT/'independent_cells.json',dict(protocol=pb,cells=independent))
    lines=['# Phase / amplitude replication on groups absent from F71','',p['boundary'],'',
           '| Contrast | Mean target-minus-fixed-wrong log M effect | Positive groups | Exploratory 95% interval |',
           '|---|---:|---:|---|']
    for k,s in summary['fixed_target_wrong_logM_gap'].items():
        lines.append(f"| {k} | {s['group_equal_mean']:.8f} | {s['positive_groups']}/{s['groups']} | {s['exploratory_bootstrap95']} |")
    text='\n'.join(lines)+'\n';report=OUT/'report.md'
    if report.exists():assert report.read_text()==text
    else:report.write_text(text)
    write(OUT/'validation.json',dict(status='PHASE_NEWGROUP_REPLICATION_JOIN_PASS',protocol=pb,
        query_validations=sources,independent_cells=bind(OUT/'independent_cells.json'),result=bind(OUT/'result.json'),report=bind(report),
        candidate_arms=len(independent),groups=len(rows),independent_M_recomputed=True,new_gpu_forwards=0,
        all_science_cells_CPU=True,no_full_C128_accuracy_claim=True))
    emit(status='PHASE_NEWGROUP_REPLICATION_JOIN_PASS',groups=len(rows),candidate_arms=len(independent))
    return 0


def main():
    global OUT
    a=argparse.ArgumentParser();a.add_argument('stage',choices=('prepare','pilot','worker','join'))
    a.add_argument('--index',type=int);a.add_argument('--budget',type=float,default=430);a.add_argument('--root',type=Path,default=OUT);args=a.parse_args()
    OUT=args.root.resolve()
    assert 0<args.budget<=430
    torch=configure()
    if args.stage=='prepare':return prepare(torch)
    if args.stage=='pilot':return pilot(torch,args.budget)
    if args.stage=='join':return join(torch)
    assert args.index in SELECTED
    return worker(args.index,args.budget,torch)


if __name__=='__main__':raise SystemExit(main())
