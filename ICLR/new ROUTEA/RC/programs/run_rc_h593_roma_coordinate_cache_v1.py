#!/usr/bin/env python3
"""Qualified exact-cache producer for the unchanged coordinate V2 experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
from rc_roma_exact_feature_cache_v1 import ExactFeatureCache, equal_tree, self_test

OUT = ROOT / 'results/rc_h593_roma_coordinate_cache_v1'
AUTH = ROOT / 'registry/rc_h593_roma_coordinate_cache_authority_v1_20260922.json'
PLAN = ROOT / 'plan/RC_H593_ROMA_EXACT_CACHE_V1_20260922.md'
LAUNCH = ROOT / 'slurm/rc_h593_roma_coordinate_cache_v1.sbatch'


def prepare():
    assert not AUTH.exists()
    sources = [Path(__file__), ROOT/'programs/rc_roma_exact_feature_cache_v1.py', PLAN, LAUNCH]
    a = dict(status='COORDINATE_EXACT_CACHE_EXECUTION_AUTHORIZED',
             scientific_authority=C.bind(C.AUTH), scientific_program=C.bind(C.__file__),
             sources=[C.bind(p) for p in sources],
             pilot_ordinal=0, candidates=128, levels=list(C.LEVELS),
             qualification='Fresh unmodified vs cached all dense output bytes; original sealed four-arm token maps, coordinates, content, C4 and 127x6 features',
             changes='Only memoize f, refiner_features, matcher forward; original match, refinement, hooks, scoring and original CPU validator unchanged',
             memory='One query and one reference feature set plus current pair matcher; clone outputs before consumers',
             provenance='Original authority continues to identify frozen scientific protocol; new parts/payload name execution authority and qualification explicitly',
             preserved='Existing qualified V2 queries and partial 16-pair files; same original downstream evaluation and fivefold',
             user_authorization='2026-09-22 user: 改, after concrete cache-reuse audit')
    C.write(AUTH, a)
    result = self_test(); result['authority'] = C.bind(AUTH)
    C.write(OUT/'preflight.json', result)
    print(result, flush=True)


def checked_authority():
    a = C.read(AUTH)
    assert a['status'] == 'COORDINATE_EXACT_CACHE_EXECUTION_AUTHORIZED'
    for b in [a['scientific_authority'], a['scientific_program'], *a['sources']]:
        C.checked(b)
    p = C.read(OUT/'preflight.json')
    assert p['status'] == 'ROMA_EXACT_CACHE_SYNTHETIC_PASS' and p['authority'] == C.bind(AUTH)
    return a


def qualify(a):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    workers = C.read(C.WORKERS)['records']; w = workers[0]
    allowed = {Path(b['path']).resolve() for kind in ('raw', 'roma') for b in w[kind].values()}
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve(); s = str(p).lower()
        assert not any(k in s for k in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi', 'formal392', '/grozi/', '/isic/', '/reports/'))
        if ROOT/'results' in p.parents:
            assert OUT in p.parents or C.OUT/'query000' in p.parents or p in allowed or p == C.WORKERS, str(p)
    sys.addaudithook(audit)
    old_validation = C.read(C.OUT/'query000/validation.json')
    assert old_validation['status'] == 'ROMA_COORDINATE_QUERY_PASS'
    old = C.read(C.checked(old_validation['payload']))
    q, prior, refs = C.load_input(w); profile = C.read(C.PROFILE); C.check_profile(profile)
    core = C.M.legacy_core(profile); model = C.M.gpu_model(profile)
    hook = C.CoordinateHook(model); cache = ExactFeatureCache(model)
    captured = {}
    def capture(module, args, output):
        captured.clear()
        captured.update({k: output[k].detach().cpu().clone() for k in ('warp_AB', 'warp_BA', 'confidence_AB', 'confidence_BA')})
    handle = model.matcher.register_forward_hook(capture)
    qpath = Path(q['query_source_path']); assert C.bind(qpath)['sha256'] == q['query_source_sha256']
    qgeom, _ = C.M.geometry(qpath, q['query_source_sha256'], 'coordinate-query', q['query_grid_shape'], q['processor_input_frame'], profile['sources']['processor']['sha256'])
    qi = core.oriented(qpath); rows = []; scores = {k: [] for k in C.LEVELS}
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    for part_index, binding in enumerate(old['parts']):
        part = torch.load(C.checked(binding), map_location='cpu', weights_only=True)
        for pair in part['pairs']:
            pos = pair['candidate_position']; ref = refs[pair['physical_row']]
            rp = Path(ref['source_path']); assert C.bind(rp)['sha256'] == ref['source_image_sha256']
            rg, _ = C.M.geometry(rp, ref['source_image_sha256'], 'coordinate-reference', ref['grid_shape'], 'DECODED_RAW_BEFORE_EXIF', profile['sources']['processor']['sha256'])
            ri = core.oriented(rp); native = None
            for name in C.LEVELS:
                timings = {}
                outputs = []
                coarse_outputs = []
                calls = []
                for enabled in (False, True):
                    cache.enabled = enabled; hook.select(name)
                    torch.cuda.synchronize(); t = time.monotonic()
                    outputs.append(model.match(qi, ri))
                    torch.cuda.synchronize(); timings['cached' if enabled else 'unmodified'] = time.monotonic() - t
                    coarse_outputs.append(dict(captured)); calls.append(list(hook.calls))
                fresh, pred = outputs
                assert equal_tree(fresh, pred), ('FULL_DENSE_OUTPUT_BYTES', pos, name)
                assert equal_tree(coarse_outputs[0], coarse_outputs[1]) and equal_tree(coarse_outputs[1], pair['coarse_matcher']), ('COARSE_BYTES', pos, name)
                assert equal_tree(calls[0], calls[1]), ('HOOK_TRACE', pos, name)
                u = core.cell_means(pred['overlap_AB'][0, ..., 0].cpu(), qgeom)
                v = core.cell_means(pred['overlap_BA'][0, ..., 0].cpu(), rg)
                c4 = C.c4(core, q['query_tokens'], ref['tokens'], u, v)
                intermediate = dict(query_coordinates=C.I.coordinate_record(pred['warp_AB'], qgeom),
                                    reference_coordinates=C.I.coordinate_record(pred['warp_BA'], rg),
                                    content=C.I.content_record(q['query_tokens'], ref['tokens'], u, v))
                sealed = pair['arms'][name]
                assert equal_tree((u, v, c4, intermediate), (sealed['query_visibility'], sealed['reference_visibility'], sealed['scores'], sealed['intermediate'])), ('SEALED_INTERMEDIATES', pos, name)
                if name == 'NATIVE':
                    native = pred
                else:
                    rms = {k: float(((pred[k] - native[k]).double() ** 2).mean().sqrt()) for k in ('warp_AB', 'warp_BA')}
                    assert rms == sealed['diagnostics']['final_warp_RMS_normalized']
                    assert calls[1] == sealed['diagnostics']['calls']
                scores[name].append(c4)
                rows.append(dict(candidate=pos, level=name, seconds=timings, dense_fields=list(pred),
                                 full_dense_bit_exact=True, sealed_intermediates_bit_exact=True))
                del fresh, outputs, coarse_outputs
                if name != 'NATIVE':
                    del pred
            del native
        C.write(OUT/f'qualification_part{part_index:02d}.json', dict(authority=C.bind(AUTH), pairs=part_index * 16 + 16, checks=rows[-64:]))
        print(dict(event='EXACT_CACHE_QUALIFICATION_PART', part=part_index, pairs=(part_index+1)*16, seconds=time.monotonic()-start), flush=True)
    raw = list(map(float, q['candidate_raw_scores'])); winner = old['winner']; challengers = old['challenger_positions']
    readout_checks = 0
    probe_head = torch.tensor([.2, .3, .1, -.2, .4, -.1], dtype=torch.float64)
    for name in C.LEVELS:
        evidence = dict(enumerate(scores[name]))
        x = torch.stack([C.candidate_feature(raw, evidence, c, winner) for c in challengers])
        previous = torch.tensor(old['modes'][name]['X'], dtype=torch.float64)
        assert equal_tree(x, previous)
        assert equal_tree(x @ probe_head + .05, previous @ probe_head + .05)
        readout_checks += 127
    counters = cache.summary()
    assert counters['counts']['f_misses'] == 129 and counters['counts']['refiner_features_misses'] == 258 and counters['counts']['matcher_misses'] == 128
    result = dict(status='ROMA_FULL128_EXACT_CACHE_PASS', authority=C.bind(AUTH), source_validation=C.bind(C.OUT/'query000/validation.json'),
                  pairs=128, levels=list(C.LEVELS), comparisons=len(rows), all_dense_outputs_bit_exact=True,
                  all_sealed_intermediates_bit_exact=True, synthetic_readout_checks=readout_checks,
                  all_127x6_features_bit_exact=True, labels_read=0, train_updates=0,
                  runtime_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(), cache=counters,
                  forward_seconds={k: sum(row['seconds'][k] for row in rows) for k in ('unmodified', 'cached')},
                  timing_scope='Interleaved fresh and cached forward only; worker CPU diagnostics, startup and verification excluded',
                  rows=rows)
    assert result['peak_gpu_bytes'] < 35 * 1024**3
    C.write(OUT/'qualification.json', result)
    C.write(OUT/'validation.json', dict(status='ROMA_FULL128_EXACT_CACHE_PASS', authority=C.bind(AUTH), payload=C.bind(OUT/'qualification.json')))
    handle.remove(); hook.close(); cache.close()
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


def worker(a, index):
    qualification = C.read(OUT/'validation.json')
    assert qualification['status'] == 'ROMA_FULL128_EXACT_CACHE_PASS' and qualification['authority'] == C.bind(AUTH)
    C.checked(qualification['payload'])
    producer = dict(execution_authority=C.bind(AUTH), qualification=C.bind(OUT/'validation.json'),
                    kind='Byte-qualified memoization under unchanged scientific coordinate V2 protocol')
    scientific, workers = C.guard('worker', index)
    folder = C.OUT/f'query{index:03d}'
    if (folder/'validation.json').exists():
        v = C.read(folder/'validation.json'); C.checked(v['payload'])
        assert v['status'] == 'ROMA_COORDINATE_QUERY_PASS' and v['authority'] == C.bind(C.AUTH)
        print(dict(event='REUSED_QUALIFIED_QUERY', index=index), flush=True)
        return
    original_factory, original_save, original_write = C.M.gpu_model, C.save, C.write
    caches = []
    def factory(profile):
        model = original_factory(profile); cache = ExactFeatureCache(model); caches.append(cache)
        return model
    def save(path, value):
        if Path(path).parent == folder and Path(path).name.startswith('part'):
            value = dict(value, execution=producer)
        return original_save(path, value)
    def write(path, value):
        if Path(path) == folder/'payload.json':
            value = dict(value, execution=producer)
        return original_write(path, value)
    C.M.gpu_model, C.save, C.write = factory, save, write
    start = time.monotonic()
    try:
        C.worker(scientific, workers[index], index)
        validation = C.read(folder/'validation.json'); C.checked(validation['payload'])
        assert validation['status'] == 'ROMA_COORDINATE_QUERY_PASS'
        original_write(folder/'exact_cache_execution.json', dict(status='QUALIFIED_EXACT_CACHE_QUERY_COMPLETE',
                       execution=producer, validation=C.bind(folder/'validation.json'), job_id=os.environ['SLURM_JOB_ID'],
                       seconds=time.monotonic()-start, caches=[c.summary() for c in caches]))
    finally:
        C.M.gpu_model, C.save, C.write = original_factory, original_save, original_write
        for cache in caches:
            cache.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare', 'qualify', 'worker')); parser.add_argument('--index', type=int)
    args = parser.parse_args(); torch.set_num_threads(8); torch.set_num_interop_threads(1)
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'qualify':
        qualify(checked_authority())
    else:
        worker(checked_authority(), args.index)
