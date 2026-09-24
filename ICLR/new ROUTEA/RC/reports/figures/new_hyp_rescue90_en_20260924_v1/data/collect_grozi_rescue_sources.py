#!/usr/bin/env python3
"""Verify and export frozen GroZi rescue evidence, without inference."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path('/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC')
WORK = ROOT / 'results/rc_new_hyp_grozi120_external_v1'
OUT = ROOT / 'reports/figures/new_hyp_rescue90_20260924_v1/data'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}

def bind(path, expected=None):
    path = Path(path).resolve()
    key = str(path)
    if key not in SOURCES:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda: f.read(8 << 20), b''):
                h.update(data)
        SOURCES[key] = dict(path=key, sha256=h.hexdigest(), bytes=path.stat().st_size)
    if expected:
        assert SOURCES[key]['sha256'] == expected, (key, 'SHA')
    return SOURCES[key]

def read(path, expected=None):
    return json.loads(Path(bind(path, expected)['path']).read_text())

result = read(WORK / 'result.json', '0bee519b388b4e54209a843199ba34285282ec95ec5941ce307b9b9165e426fa')
for filename in ['result_validation.json', 'independent_final_audit_v1.json']:
    validation = read(WORK / filename)
    assert 'PASS' in validation['status']
    bind(validation['result']['path'], validation['result']['sha256'])
authority = read(result['authority']['path'], result['authority']['sha256'])
head_source = authority['heads']['COST1']['head']
read(head_source['path'], head_source['sha256'])
assert 'PASS' in read(authority['heads']['COST1']['validation']['path'],
                       authority['heads']['COST1']['validation']['sha256'])['status']
workers = {r['query_id']: r for r in read(WORK / 'worker_manifest.json')['records']}
gallery = {r['identity']: r for r in read(WORK / 'gallery_append_manifest.json')['records']}
eligible = sorted([r for r in result['rows'] if not r['correct']['RAW'] and r['correct']['COST1']], key=lambda r:r['query_id'])
assert len(eligible) == 34
seen = set()
selected, repeats = [], []
for r in eligible:
    if r['identity'] not in seen:
        selected.append(r)
        seen.add(r['identity'])
    else:
        repeats.append(r)
selected += repeats[:30-len(selected)]
selected.sort(key=lambda r:r['query_id'])
assert len(selected) == len({r['query_id'] for r in selected}) == 30
assert len({r['identity'] for r in selected}) == 23
rule = 'First eligible query_id per identity, then earliest remaining query_ids to 30; display sorted by query_id; no selection on maps, S, M or logit magnitude.'
cases = []
by_shard = collections.defaultdict(list)
for r in selected:
    by_shard[workers[r['query_id']]['execution_ordinal'] // 8].append(r)
for shard, rows in sorted(by_shard.items()):
    payloads = {}
    for stage in ['raw', 'roma']:
        folder = WORK / stage / f'shard{shard:02d}'
        receipt, v = read(folder / 'receipt.json'), read(folder / 'validation.json')
        assert 'PASS' in v['status'] and receipt['payload'] == v['payload']
        binding = bind(receipt['payload']['path'], receipt['payload']['sha256'])
        payloads[stage] = torch.load(binding['path'], map_location='cpu', mmap=True, weights_only=True)
    prediction_folder = WORK / 'predictions' / f'shard{shard:02d}'
    pv = read(prediction_folder / 'validation.json')
    assert 'PASS' in pv['status']
    pred = read(pv['payload']['path'], pv['payload']['sha256'])
    assert pred['target_reads'] == pred['training_updates'] == 0
    for stage in ['raw', 'roma']:
        assert pred[stage]['sha256'] == bind(WORK / stage / f'shard{shard:02d}' / 'payload.pt')['sha256']
    raw, roma = payloads['raw'], payloads['roma']
    for r in rows:
        qid = r['query_id']; worker = workers[qid]
        q = next(z for z in raw['records'] if z['query_id'] == qid)
        m = next(z for z in roma['records'] if z['query_id'] == qid)
        p = next(z for z in pred['records'] if z['query_id'] == qid)
        assert worker['execution_ordinal'] == q['execution_ordinal'] == m['execution_ordinal'] == p['execution_ordinal']
        assert q['query_tokens_sha256'] == m['query_tokens_sha256']
        assert q['query_grid_shape'] == m['query_grid_shape']
        assert q['candidate_physical_rows'] == m['candidate_physical_rows']
        assert r['target_in_C128'] and not r['holds']['COST1']
        target, wrong = gallery[r['identity']]['physical_row'], r['selected']['RAW']
        assert target == r['selected']['COST1'] == p['models']['COST1']['selected'] != wrong
        axis = q['candidate_physical_rows']
        winner = axis.index(q['candidate_ranked_physical_rows'][0])
        assert axis[winner] == wrong == p['raw_selected']
        assert int(q['candidate_raw_scores'].argmax()) == winner
        challenger_positions = [i for i in range(128) if i != winner]
        logits = np.array([float.fromhex(v) for v in p['models']['COST1']['logits_hex']])
        assert len(logits) == len(challenger_positions) == 127
        assert logits.max() > 0 and axis[challenger_positions[int(logits.argmax())]] == target
        bind(worker['image_path'], m['query_source_sha256'])
        evidence = {}
        for role, physical in [('wrong', wrong), ('target', target)]:
            c = next(z for z in m['candidates'] if z['physical_row'] == physical)
            ref = raw['references'][physical]
            assert c['reference_tokens_sha256'] == ref['tokens_sha256']
            assert c['reference_grid_shape'] == ref['grid_shape']
            bind(ref['source_path'], c['reference_image_sha256'])
            for side, grid in [('query', m['query_grid_shape']), ('reference', c['reference_grid_shape'])]:
                a = c[side + '_visibility'].numpy()
                assert a.dtype == np.float64 and np.prod(grid) == a.size and np.isfinite(a).all()
                assert ((a >= 0) & (a <= 1)).all()
                assert hashlib.sha256(a.tobytes()).hexdigest() == c[side + '_map_sha256']
            mass = float(np.sqrt(c['query_visibility'].numpy().mean() * c['reference_visibility'].numpy().mean()))
            assert abs(mass - c['old_scores']['visibility_mass']) < 1e-14
            evidence[role] = dict(physical_row=physical, reference_path=ref['source_path'],
                reference_image_sha256=c['reference_image_sha256'], reference_grid=ref['grid_shape'],
                reference_geometry=c['reference_geometry'], scores=c['old_scores'],
                query_map_sha256=c['query_map_sha256'], reference_map_sha256=c['reference_map_sha256'],
                query_visibility=c['query_visibility'].tolist(), reference_visibility=c['reference_visibility'].tolist(),
                logit=0.0 if role == 'wrong' else float(logits.max()),
                logit_role='HOLD baseline utility; not an evaluated challenger logit' if role == 'wrong' else 'sealed COST1 target challenger logit')
        cases.append(dict(dataset='grozi', dataset_label='GroZi 商品', label='GroZi 商品', panel='GroZi480 外部确认面板',
            model='COST1', model_lineage=result['model_lineage'], evidence_scope='外部确认结果的事后可视化',
            query_id=qid, display_id=qid, source_query_id=qid, execution_ordinal=worker['execution_ordinal'],
            identity=r['identity'], query_path=worker['image_path'], source_result=str(WORK / 'result.json'),
            raw_correct=False, final_correct=True, raw_selected=wrong, final_selected=target,
            target_physical=target, wrong_physical=wrong, wrong_role='RAW 错误 reference',
            raw_root=str(WORK / 'raw'), roma_root=str(WORK / 'roma'),
            raw_payload=str(WORK / 'raw' / f'shard{shard:02d}' / 'payload.pt'),
            roma_payload=str(WORK / 'roma' / f'shard{shard:02d}' / 'payload.pt'),
            prediction_payload=str(prediction_folder / 'payload.json'), head_source=head_source,
            query_grid=m['query_grid_shape'], query_geometry=m['query_geometry'],
            processor_input_frame=m['processor_input_frame'], evidence=evidence,
            sealed_COST1_logits_hex=p['models']['COST1']['logits_hex'],
            challenger_physical_rows=[axis[i] for i in challenger_positions],
            selected_logit=float(logits.max()), maximum_logit=float(logits.max()), raw_hold_utility=0.0,
            action='SWITCH', candidate_source=result['candidate_source'], source_video=r['source_video'],
            source_frame=r['source_frame'], component=r['component'], RAW_target_rank=r['ranks']['RAW'],
            selection_rule=rule, inference_calls=0, training_updates=0))
    print('VERIFIED', shard, [r['query_id'] for r in rows], flush=True)
    del raw, roma, payloads
cases.sort(key=lambda c:c['query_id'])
output = dict(status='GROZI_SELECTED30_FROZEN_RESCUE_SOURCES_PASS', selected=cases, selected30=cases,
    eligible_count=len(eligible), eligible_identity_count=len({r['identity'] for r in eligible}),
    selected_count=len(cases), selected_identity_count=len({r['identity'] for r in cases}),
    eligible_query_ids=[r['query_id'] for r in eligible],
    excluded_query_ids=[r['query_id'] for r in eligible if r['query_id'] not in {c['query_id'] for c in cases}],
    selected_source_video_count=len({c['source_video'] for c in cases}),
    complete_panel=dict(n=480, RAW_correct=321, COST1_correct=355, rescue=34, breaks=0,
                        candidate_source=result['candidate_source'], scope=result['GO_scope']),
    selection_rule=rule, sources=list(SOURCES.values()), inference_calls=0, training_updates=0)
(OUT / 'grozi_candidates.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print('DONE', len(cases), 'identities', output['selected_identity_count'], 'videos', output['selected_source_video_count'])
