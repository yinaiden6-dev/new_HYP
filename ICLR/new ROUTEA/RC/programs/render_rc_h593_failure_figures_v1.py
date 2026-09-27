#!/usr/bin/env python3
"""All sealed H593 COST1 errors, using cached maps and the rescue90 presentation style."""
from __future__ import annotations
import argparse
import collections
import csv
import functools
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/h593_failure_figures_mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_pdf import PdfPages
import render_rc_new_hyp_visibility_cases_v1 as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/figures/new_hyp_h593_failures_en_20260927_v1'
DATA = ROOT / 'results/rc_new_hyp593_oof5_v1'
FIT = ROOT / 'results/rc_six_cause_isolation_v1/loss_binding'
GALLERY = ROOT.parents[2] / 'dailymed/data/box_flat_20000_images/data/raw_images'
KIND = {'WRONG_HOLD': 'Uncorrected: wrong HOLD', 'WRONG_SWITCH': 'Wrong-to-wrong SWITCH',
        'BREAK': 'Break: RAW correct, COST1 wrong'}


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def bind(p):
    p = V.source(p)
    return {'path': str(p.resolve()), 'sha256': V.sha(p)}


@functools.lru_cache(maxsize=3)
def cached(folder):
    folder = Path(folder)
    receipt = V.read(folder / 'receipt.json')
    validation = V.read(folder / 'validation.json')
    assert 'PASS' in validation['status'] and validation['payload'] == receipt['payload']
    p = V.source(receipt['payload']['path'], receipt['payload']['sha256'])
    return torch.load(p, map_location='cpu', weights_only=True, mmap=True), receipt['payload']


def gallery():
    sys.path.insert(0, str(ROOT / 'src'))
    from rc_aslo_xf.gallery_identity_repair import build_identity_map
    paths = []
    for directory, subdirs, names in os.walk(GALLERY):
        subdirs.sort()
        paths += [Path(directory) / n for n in sorted(names)
                  if Path(n).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
                  and (Path(directory) / n).is_file()]
    assert len(paths) == 5413
    mapping = build_identity_map([p.stem.strip() for p in paths])
    assert len(set(mapping.labels)) == 5412
    V.source(ROOT / 'registry/gallery_identity_repair_v1.json')
    return paths, list(mapping.labels), mapping.corrected_row_identity_mapping_sha256


def inputs():
    result = V.read(FIT / 'result.json')
    valid = V.read(FIT / 'validation.json')
    assert valid['status'] == 'SIX_CAUSE_LOSS_BINDING_PARITY_COUNTS_PASS'
    assert valid['result'] == bind(FIT / 'result.json')
    assert len(result['rows']) == 593 and sum(r['correct']['COST1'] for r in result['rows']) == 481
    rows = sorted([r for r in result['rows'] if not r['correct']['COST1']],
                  key=lambda r: (not r['original_query_id'].startswith(('DIFFICULT-', 'NDV2-')), r['original_query_id']))
    assert len(rows) == 112
    roles = {r['query_id']: r for r in V.read(DATA / 'metadata/curator_roles.json')['records']}
    workers = V.read(DATA / 'metadata/worker_manifest.json')['records']
    locations = {}
    for lane in ('reuse', 'missing'):
        for i, w in enumerate(w for w in workers if ('reuse' in w) == (lane == 'reuse')):
            locations[w['query_id']] = (lane, i // 8)
    heads = {}
    for fold in range(5):
        val = V.read(FIT / f'fold{fold}' / 'validation.json')
        assert val['status'] == 'SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS'
        p = V.source(val['payload']['path'], val['payload']['sha256'])
        head = V.read(p)
        heads[fold] = (head, val['payload'], {r['query_id']: r['models']['COST1'] for r in head['predictions']})
    return rows, roles, {r['query_id']: r for r in workers}, locations, heads


def extract(row, number, role, worker, location, heads, paths, labels):
    qid = row['query_id']; lane, shard = location
    fd, fb = cached(DATA / 'features' / lane / f'shard{shard:02d}')
    f = next(r for r in fd['records'] if r['query_id'] == qid)
    head, hb, preds = heads[row['fold']]; pred = preds[qid]
    assert qid not in head['train_query_ids'] and head['fold'] == role['outer_fold'] == row['fold']
    theta = np.array([float.fromhex(x) for x in head['parameters']['COST1']])
    X = f['modes']['REAL']['X'].numpy()
    logits = np.array([float.fromhex(x) for x in pred['logits_hex']], dtype=np.float64)
    error = float(np.max(np.abs((X * theta[:-1]).sum(axis=1) + theta[-1] - logits)))
    assert error < 2e-10 and X.shape == (127, 6)
    axis = f['candidate_physical_rows']; raw = axis[f['winner']]
    challengers = [axis[p] for p in f['challenger_positions']]
    chosen = challengers[int(logits.argmax())] if logits.max() > 0 else raw
    assert chosen == pred['selected'] and labels[chosen] != role['identity']
    assert (labels[raw] == role['identity']) == row['correct']['RAW']
    targets = [p for p in f['raw_ranked_physical_rows'] if labels[p] == role['identity']]
    assert len(targets) == 1
    target = targets[0]; present = target in axis
    assert present == row['target_in_C128']
    if lane == 'reuse':
        reuse = worker['reuse']; roma_folder = Path(reuse['roma']['payload']['path']).parent
        source_qid = reuse['source_query_id']
    else:
        roma_folder = DATA / 'roma' / f'shard{shard:02d}'; source_qid = qid
    roma, rb = cached(roma_folder)
    m = next(r for r in roma['records'] if r['query_id'] == source_qid)
    assert axis == m['candidate_physical_rows'] and axis[int(m['candidate_raw_scores'].argmax())] == raw
    assert f['source_image_sha256'] == worker['source_image_sha256'] == m['query_source_sha256']
    query_path = V.source(worker['query_image_path'], m['query_source_sha256'])
    photos = {'query': V.encoded_photo(query_path, m['processor_input_frame'], m['query_geometry'])}
    arrays = {'decision_X': X.copy()}; evidence = {}
    for name, physical in [('wrong', chosen), ('target', target)]:
        rp = paths[physical]
        cand = next((c for c in m['candidates'] if c['physical_row'] == physical), None)
        if cand is None:
            assert name == 'target' and not present
            V.source(rp)
            with Image.open(rp) as im:
                photos[name] = im.convert('RGB'); photos[name].thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            evidence[name] = {'physical_row': physical, 'identity': labels[physical],
                'reference_path': str(rp), 'reference_image_sha256': V.sha(rp), 'scores': None,
                'visibility_available': False, 'reason': 'Target excluded by natural C128; no cached pair was computed',
                'display_frame': 'DECODED_RAW_GALLERY_PHOTO_NOT_PAIR_INPUT'}
            continue
        V.source(rp, cand['reference_image_sha256'])
        geom = cand['reference_geometry']
        photos[name] = V.encoded_photo(rp, geom['processor_input_frame'], geom)
        for side, grid in [('query', m['query_grid_shape']), ('reference', cand['reference_grid_shape'])]:
            a = cand[side + '_visibility'].numpy()
            assert a.dtype == np.float64 and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
            assert hashlib.sha256(a.tobytes()).hexdigest() == cand[side + '_map_sha256']
            arrays[f'{name}_{side}_visibility'] = a.reshape(grid).copy()
        mass = math.sqrt(float(cand['query_visibility'].mean()) * float(cand['reference_visibility'].mean()))
        assert abs(mass - cand['old_scores']['visibility_mass']) < 1e-14
        evidence[name] = {'physical_row': physical, 'identity': labels[physical], 'reference_path': str(rp),
            'reference_image_sha256': cand['reference_image_sha256'], 'reference_geometry': geom,
            'query_map_sha256': cand['query_map_sha256'], 'reference_map_sha256': cand['reference_map_sha256'],
            'scores': cand['old_scores'], 'visibility_available': True}
    kind = 'BREAK' if row['correct']['RAW'] else ('WRONG_HOLD' if chosen == raw else 'WRONG_SWITCH')
    stem = f"{number:03d}_COST1_{row['original_query_id']}"
    score_axis = [raw] + challengers; scores_hex = [0.0.hex()] + pred['logits_hex']
    case = {'number': number, 'stem': stem, 'query_id': qid, 'display_id': row['original_query_id'],
        'identity': role['identity'], 'track': role['track'], 'dataset': 'medicine', 'model': 'COST1',
        'model_lineage': 'H593 original grouped OOF NATIVE7-COST1; same medicine source as rescue90',
        'source_fold': row['fold'], 'component': row['component'], 'source_query_id': source_qid,
        'raw_correct': row['correct']['RAW'], 'final_correct': False, 'raw_selected': raw,
        'final_selected': chosen, 'wrong_physical': chosen, 'target_physical': target,
        'target_in_C128': present, 'target_raw_full_gallery_rank': f['raw_ranked_physical_rows'].index(target) + 1,
        'failure_kind': kind, 'action': 'HOLD' if raw == chosen else 'SWITCH',
        'query_path': str(query_path), 'query_source_sha256': m['query_source_sha256'],
        'query_grid': m['query_grid_shape'], 'query_geometry': m['query_geometry'],
        'processor_input_frame': m['processor_input_frame'], 'evidence': evidence,
        'head_source': hb, 'head_parameters_hex': head['parameters']['COST1'],
        'sealed_COST1_logits_hex': pred['logits_hex'], 'challenger_physical_rows': challengers,
        'scores128_physical_axis': score_axis, 'scores128_hex': scores_hex,
        'candidate_physical_rows': axis, 'raw_candidate_scores_hex': [float(x).hex() for x in m['candidate_raw_scores']],
        'selected_logit': 0.0 if chosen == raw else float(logits.max()),
        'max_challenger_logit': float(logits.max()),
        'target_logit': float.fromhex(scores_hex[score_axis.index(target)]) if present else None,
        'feature_payload': fb, 'roma_payload': rb, 'decision_logit_error': error,
        'query_excluded_from_head_training': True, 'fold_number_displayed': False,
        'selection_rule': 'All 112 incorrect COST1 predictions among H593; difficult/new difficult displayed first',
        'map_rendering': 'Native cached FP64 grids; nearest; fixed [0,1]; absent target maps omitted',
        'training_updates': 0, 'encoder_or_matcher_inference_calls': 0}
    return case, photos, arrays


def draw(c, photos, arrays):
    fig = V.plt.figure(figsize=(16, 9), facecolor=V.BG)
    fig.text(.05, .947, 'new HYP  /  H593 FAILURE CASES', color=V.TEAL, fontsize=10, weight='bold')
    fig.text(.05, .887, 'When reference-conditioned retrieval fails', fontsize=26, color=V.INK, weight='bold')
    subtitle = f"{c['display_id']}  |  H593 development panel  |  COST1  |  {KIND[c['failure_kind']]}"
    fig.text(.05, .837, subtitle, fontsize=12.5, color=V.MUTED)
    fig.add_artist(V.plt.Line2D([.05, .95], [.809, .809], transform=fig.transFigure, color='#DCE5ED', lw=1))
    ax = fig.add_axes([.055, .34, .26, .415]); ax.imshow(photos['query']); ax.axis('off')
    ax.set_title('Query (complete input image)', fontsize=13, color=V.INK, pad=12)
    heat = None
    for j, name in enumerate(['wrong', 'target']):
        x = .385 + j * .30; ax = fig.add_axes([x, .455, .235, .30])
        available = c['evidence'][name]['visibility_available']
        if available:
            values = arrays[f'{name}_query_visibility']
            im = ax.imshow(values, cmap='viridis', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
            assert im.get_array().tobytes() == values.tobytes()
            heat = im
            ax.set_xlabel('Query token column', fontsize=10); ax.set_ylabel('Query token row', fontsize=10)
            ax.tick_params(labelsize=9, length=3)
            for spine in ax.spines.values(): spine.set_visible(False)
            sc = c['evidence'][name]['scores']
            fig.text(x, .383, f"S = {sc['real_score']:.5f}    M = {sc['visibility_mass']:.5f}", fontsize=12, color=V.INK)
        else:
            ax.set_facecolor('#EDF1F5'); ax.set_xticks([]); ax.set_yticks([])
            ax.text(.5, .54, 'No cached target-pair map', ha='center', va='center', fontsize=14, color=V.INK, transform=ax.transAxes)
            ax.text(.5, .34, 'Correct reference is outside C128\nThis pair was not scored', ha='center', va='center', fontsize=11, color=V.MUTED, transform=ax.transAxes)
            fig.text(x, .383, 'S = not computed    M = not computed', fontsize=10.5, color=V.MUTED)
        ax.set_title('Paired with final incorrect reference' if name == 'wrong' else 'Paired with the correct reference', fontsize=11.5, pad=12)
        refax = fig.add_axes([x, .05, .235, .235]); refax.imshow(photos[name]); refax.axis('off')
        label = 'Final incorrect reference' if name == 'wrong' else 'Correct reference'
        physical = c['wrong_physical'] if name == 'wrong' else c['target_physical']
        fig.text(x + .1175, .303, f'{label} · entry {physical}', ha='center', fontsize=10.5, color=V.INK)
    cbax = fig.add_axes([.405, .344, .485, .012])
    cb = fig.colorbar(heat, cax=cbax, orientation='horizontal', ticks=[0, .5, 1]); cb.ax.tick_params(labelsize=9, length=3)
    cb.outline.set_edgecolor('#DCE5ED')
    transition = 'RAW correct → COST1 incorrect' if c['raw_correct'] else 'RAW incorrect → COST1 incorrect'
    fig.text(.055, .281, transition, fontsize=12, color=V.RED, weight='bold')
    pool = 'Target inside C128' if c['target_in_C128'] else 'Target outside C128 (retrieval miss)'
    fig.text(.055, .238, f"Decision: {c['action']}   |   {pool}\nRAW entry {c['raw_selected']} → final entry {c['final_selected']}", fontsize=9.5, color=V.MUTED, va='top', linespacing=1.6)
    t = f"{c['target_logit']:.4f}" if c['target_logit'] is not None else 'not scored'
    fig.text(.055, .155, f"Action utility: final {c['selected_logit']:.4f}; target {t}\nS: weighted content; M: joint visibility mass\nHOLD utility = 0. The head uses multiple signals.", fontsize=9, color=V.MUTED, va='top', linespacing=1.6)
    fig.text(.05, .027, 'All H593 COST1 errors  |  Actual cached token weights  |  Shared 0–1 scale  |  Not a segmentation mask', fontsize=8, color=V.MUTED)
    fig.text(.95, .027, f"{c['number']:03d}", ha='right', fontsize=10, color=V.TEAL)
    texts = [t.get_text() for t in fig.texts] + [a.get_title() for a in fig.axes]
    assert not any(re.search('[\u3400-\u9fff]', s) or 'fold' in s.lower() for s in texts)
    return fig, texts


def package(cases):
    counts = dict(collections.Counter(c['failure_kind'] for c in cases))
    assert counts == {'WRONG_HOLD': 92, 'BREAK': 3, 'WRONG_SWITCH': 17}
    assert sum(c['target_in_C128'] for c in cases) == 89
    fields = ['number', 'display_id', 'query_id', 'track', 'identity', 'failure_kind', 'action',
              'raw_correct', 'final_correct', 'raw_selected', 'final_selected', 'target_physical',
              'target_in_C128', 'target_raw_full_gallery_rank', 'selected_logit', 'target_logit', 'stem']
    with (OUT / 'data/cases.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: c[k] for k in fields} for c in cases)
    dump(OUT / 'cases_manifest.json', cases)
    font = ImageFont.truetype(V.font_manager.findfont('DejaVu Sans'), 23)
    for start in range(0, len(cases), 24):
        group = cases[start:start+24]; sheet = Image.new('RGB', (1920, 70 + math.ceil(len(group) / 3) * 390), V.BG)
        d = ImageDraw.Draw(sheet); d.text((25, 20), f'H593 COST1 failures | cases {start+1}–{start+len(group)}', font=font, fill=V.INK)
        for i, c in enumerate(group):
            with Image.open(OUT / 'medicine' / (c['stem'] + '.png')) as im:
                thumb = im.resize((640, 360), Image.Resampling.LANCZOS)
            x, y = (i % 3) * 640, (i // 3) * 390 + 70; sheet.paste(thumb, (x, y))
            d.text((x+12, y+362), c['display_id'] + ' | ' + c['failure_kind'], font=font, fill=V.INK)
        sheet.save(OUT / f'contact_sheet_{start//24+1:02d}.jpg', quality=93)
    cards=[]; md=[]
    for c in cases:
        stem=c['stem']; name=html.escape(c['display_id']); absent=not c['target_in_C128']
        cards.append(f'<article data-kind="{c["failure_kind"]}" data-pool="{"absent" if absent else "present"}"><h2>{c["number"]:03d} · {name} · {KIND[c["failure_kind"]]}</h2><a href="medicine/{stem}.png"><img src="medicine/{stem}.png" loading="lazy" alt="{name}"></a><p><a href="medicine/{stem}.png">PNG</a> · <a href="medicine/{stem}.pdf">PDF</a> · <a href="medicine/{stem}.svg">SVG</a> · <a href="data/{stem}.json">Decision data</a> · {"Target outside C128" if absent else "Target inside C128"}</p></article>')
        md.append(f'| {c["number"]} | [{c["display_id"]}](medicine/{stem}.png) | {c["failure_kind"]} | {"outside" if absent else "inside"} | [data](data/{stem}.json) |')
    header='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H593 COST1: all 112 incorrect retrievals</title>
<style>body{max-width:1400px;margin:30px auto;padding:0 20px;font:17px/1.6 system-ui;background:#fafcfe;color:#142b42}a{color:#087f8c}article{margin:28px 0;border:1px solid #dce5ed;border-radius:10px;padding:16px}img{width:100%;height:auto}button{margin:6px;padding:8px 14px;font:inherit}article[hidden]{display:none}</style>
<h1>H593 COST1: all 112 incorrect retrievals</h1><p>Same original grouped held-out COST1 model as the medicine cases in rescue90. RAW 426/593; COST1 481/593. All failures are included.</p>
<p>92 wrong HOLD · 17 wrong-to-wrong SWITCH · 3 breaks. Correct reference: 89 inside C128, 23 outside. Missing target heatmaps are explicitly unavailable, not zero maps.</p>
<p><a href="failures112.pdf">112-page PDF</a> · <a href="data/cases.csv">CSV</a> · <a href="README.md">Provenance</a></p>
<nav><button onclick="show('all')">All 112</button><button onclick="show('WRONG_HOLD')">Wrong HOLD 92</button><button onclick="show('WRONG_SWITCH')">Wrong SWITCH 17</button><button onclick="show('BREAK')">Breaks 3</button><button onclick="show('present')">Target inside 89</button><button onclick="show('absent')">Target outside 23</button></nav>'''
    (OUT/'index.html').write_text(header+'\n'.join(cards)+"<script>function show(k){document.querySelectorAll('article').forEach(x=>x.hidden=k!=='all'&&x.dataset.kind!==k&&x.dataset.pool!==k)}</script></html>")
    (OUT/'README.md').write_text('''# H593: all COST1 retrieval failures — English figures

All **112 incorrect predictions** from the original H593 grouped held-out COST1 model, exactly the medicine-model source used in the 90 successful-correction figures. RAW is 426/593 and COST1 is 481/593. This is the original external seven-parameter COST1, not the post-LLM model.

- 92 wrong HOLD, 17 wrong-to-wrong SWITCH, and 3 RAW-correct breaks.
- 89 have the correct reference in natural C128; 23 are candidate-recall misses. These 23 remain included. Their correct gallery photograph is shown, but no target-pair map or score was computed in the original C128 experiment. The empty panel is marked unavailable, never filled with invented weights.
- 9 DIFFICULT, 1 NDV2 and 102 OUTCOME cases; difficult/new difficult appear first. Every sealed failure is included, without selection by appearance or heatmap brightness.
- English 3200×1800 PNGs, SVGs, single-page PDFs, [112-page PDF](failures112.pdf), contact sheets, [offline HTML gallery](index.html) and [CSV index](data/cases.csv). References retain the enlarged full-column layout. Fold numbers appear only in provenance data.

The figure compares the **final incorrect reference** against the correct reference. The RAW winner and final action are separately labeled; wrong-to-wrong switches do not substitute the old RAW reference for the actual final error. Correct identity is determined by the original repaired gallery labels. Heatmaps are literal cached FP64 query-token weights, nearest-pixel display and fixed [0,1]. They are not segmentation masks.

Per-case JSON keeps the complete natural candidate axis, all 127 sealed challenger logits plus HOLD=0, original COST1 parameters, source fold and training exclusion, action replay and source hashes. NPZ files keep native query/reference maps for available pairs and the 127×6 decision feature matrix, not model embeddings. Rendering runs no encoder, matcher or training. Original experiments and the successful rescue figures are unchanged.

Source: `results/rc_six_cause_isolation_v1/loss_binding/result.json` and its five original fold payloads; H593 feature and RoMa caches provide the actual intermediate evidence. [Source bindings](source_manifest.json) and [validation](validation.json) document verification. Source photos are embedded in panels; original photos and large feature caches are separately stored and not copied into the Git figure directory.

## All cases

| No. | Query / PNG | Final error | Target in C128 | Evidence |
|---:|---|---|---|---|
'''+'\n'.join(md)+'\n')
    dump(OUT/'validation.json', {'status':'ALL112_COST1_FAILURE_FIGURES_RENDER_AND_REPLAY_PASS',
         'cases':len(cases), 'all_H593_failures_included':True, 'failure_kinds':counts,
         'target_in_C128':89, 'target_absent':23, 'native_maps':sum(4 if c['target_in_C128'] else 2 for c in cases),
         'max_decision_logit_replay_error':max(c['decision_logit_error'] for c in cases),
         'source_fold_hidden_in_figures':True, 'image_size':[3200,1800],
         'missing_target_maps_not_synthesized':True,'training_updates':0,'encoder_or_matcher_inference_calls':0})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args()
    torch.set_num_threads(2)
    for folder in ['medicine','data','source']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    rows, roles, workers, locations, heads=inputs()
    paths, labels, mapping=gallery()
    dump(OUT/'data/selection.json', {'all_selected_query_ids':[r['query_id'] for r in rows],
         'source_result':bind(FIT/'result.json'),'gallery_mapping_sha256':mapping,
         'gallery_paths':list(map(str,paths)), 'gallery_labels':labels,
         'selection':'All H593 COST1 errors; DIFFICULT/NDV2 first, then original query ID'})
    cases=[]; captions={}
    with PdfPages(OUT/('preview.pdf' if args.limit else 'failures112.pdf')) as pdf:
        for i,row in enumerate(rows[:args.limit] if args.limit else rows,1):
            qid=row['query_id'];c,photos,arrays=extract(row,i,roles[qid],workers[qid],locations[qid],heads,paths,labels)
            c['source_result']=bind(FIT/'result.json')
            np.savez_compressed(OUT/'data'/(c['stem']+'.npz'),**arrays)
            with np.load(OUT/'data'/(c['stem']+'.npz')) as saved:
                for k,a in arrays.items():assert saved[k].tobytes()==a.tobytes() and saved[k].shape==a.shape
            dump(OUT/'data'/(c['stem']+'.json'),c)
            fig,texts=draw(c,photos,arrays)
            for ext in ['png','pdf','svg']:fig.savefig(OUT/'medicine'/(c['stem']+'.'+ext),dpi=200)
            pdf.savefig(fig);V.plt.close(fig)
            with Image.open(OUT/'medicine'/(c['stem']+'.png')) as im:assert im.size==(3200,1800)
            cases.append(c);captions[c['stem']]=texts
            print(json.dumps({'rendered':i,'total':len(rows),'query':c['display_id'],'kind':c['failure_kind'],'target_in_C128':c['target_in_C128']}),flush=True)
    if args.limit:return
    package(cases)
    dump(OUT/'data/rendered_english_text.json',captions)
    for p in [Path(__file__),Path(V.__file__)]:
        V.source(p);shutil.copy2(p,OUT/'source'/p.name)
    dump(OUT/'source_manifest.json',V.SOURCES)
    print(json.dumps({'status':'COMPLETE','cases':len(cases),'output':str(OUT)}),flush=True)


if __name__=='__main__':main()
