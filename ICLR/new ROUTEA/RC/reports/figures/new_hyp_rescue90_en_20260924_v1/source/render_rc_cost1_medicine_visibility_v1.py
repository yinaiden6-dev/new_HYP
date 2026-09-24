#!/usr/bin/env python3
"""Render existing COST1 medicine decisions and literal cached visibility grids."""
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/new_hyp_cost1_medicine_mpl')
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont
import render_rc_new_hyp_visibility_cases_v1 as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/figures/new_hyp_cost1_medicine_visibility_20260919_v1'
DATA = ROOT / 'results/rc_new_hyp593_oof5_v1'
FIT = ROOT / 'results/rc_six_cause_isolation_v1/loss_binding'
SELECTION = ('OUTCOME-0438 first; then first RAW-to-COST1 rescue per previously '
             'unseen identity, ordered by original_query_id, until 12 cases. '
             'Selection does not inspect visibility maps.')


def read(path):
    return V.read(path)


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=V.sha(path))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def cache(folder):
    receipt = read(folder / 'receipt.json')
    validation = read(folder / 'validation.json')
    assert 'PASS' in validation['status']
    assert validation['payload'] == receipt['payload']
    p = V.source(receipt['payload']['path'], receipt['payload']['sha256'])
    return torch.load(p, map_location='cpu', weights_only=True, mmap=True)


def select():
    result = read(FIT / 'result.json')
    val = read(FIT / 'validation.json')
    assert val['status'] == 'SIX_CAUSE_LOSS_BINDING_PARITY_COUNTS_PASS'
    assert val['result'] == bind(FIT / 'result.json')
    roles = {r['query_id']: r for r in read(DATA / 'metadata/curator_roles.json')['records']}
    workers = read(DATA / 'metadata/worker_manifest.json')['records']
    locations = {}
    for lane in ('reuse', 'missing'):
        ws = [w for w in workers if ('reuse' in w) == (lane == 'reuse')]
        for i, w in enumerate(ws):
            locations[w['query_id']] = (lane, i // 8)
    workers = {w['query_id']: w for w in workers}
    rescues = sorted((r for r in result['rows'] if not r['correct']['RAW'] and r['correct']['COST1']),
                     key=lambda r: r['original_query_id'])
    assert len(rescues) == 58
    selected = [next(r for r in rescues if r['original_query_id'] == 'OUTCOME-0438')]
    identities = {roles[selected[0]['query_id']]['identity']}
    for r in rescues:
        identity = roles[r['query_id']]['identity']
        if identity not in identities:
            selected.append(r)
            identities.add(identity)
        if len(selected) == 12:
            break
    assert len(selected) == len(identities) == 12
    return selected, roles, workers, locations


def extract(row, role, worker, location, stem):
    qid = row['query_id']
    folder = FIT / f"fold{row['fold']}"
    validation = read(folder / 'validation.json')
    assert validation['status'] == 'SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS'
    p = V.source(validation['payload']['path'], validation['payload']['sha256'])
    fitted = read(p)
    assert fitted['fold'] == role['outer_fold'] == row['fold']
    assert qid not in fitted['train_query_ids']
    prediction = next(p for p in fitted['predictions'] if p['query_id'] == qid)['models']['COST1']
    lane, shard = location
    features = cache(DATA / 'features' / lane / f'shard{shard:02d}')
    f = next(r for r in features['records'] if r['query_id'] == qid)
    assert f['source_image_sha256'] == worker['source_image_sha256']
    theta = np.array([float.fromhex(v) for v in fitted['parameters']['COST1']])
    logits = np.sum(f['modes']['REAL']['X'].numpy() * theta[:-1], axis=1) + theta[-1]
    sealed = np.array([float.fromhex(v) for v in prediction['logits_hex']])
    error = float(np.max(np.abs(logits - sealed)))
    assert error < 2e-10
    axis = f['candidate_physical_rows']
    selected = axis[f['challenger_positions'][int(logits.argmax())]] if logits.max() > 0 else axis[f['winner']]
    assert selected == prediction['selected'] and row['correct']['COST1']
    target = selected
    wrong = axis[f['winner']]
    assert target != wrong and row['target_in_C128']
    if lane == 'reuse':
        reuse = worker['reuse']
        rawfolder = Path(reuse['token_raw']['payload']['path']).parent
        romafolder = Path(reuse['roma']['payload']['path']).parent
        source_qid = reuse['source_query_id']
    else:
        rawfolder = DATA / 'raw' / f'shard{shard:02d}'
        romafolder = DATA / 'roma' / f'shard{shard:02d}'
        source_qid = qid
    raw, roma = cache(rawfolder), cache(romafolder)
    q = next(r for r in raw['records'] if r['query_id'] == source_qid)
    m = next(r for r in roma['records'] if r['query_id'] == source_qid)
    assert axis == q['candidate_physical_rows'] == m['candidate_physical_rows']
    assert q['query_tokens_sha256'] == m['query_tokens_sha256']
    assert q['query_grid_shape'] == m['query_grid_shape']
    assert axis[int(q['candidate_raw_scores'].argmax())] == wrong
    query_path = V.source(q['query_source_path'], m['query_source_sha256'])
    photos = {'query': V.encoded_photo(query_path, m['processor_input_frame'], m['query_geometry'])}
    arrays, evidence = {}, {}
    for name, physical in [('wrong', wrong), ('target', target)]:
        candidate = next(c for c in m['candidates'] if c['physical_row'] == physical)
        ref = raw['references'][physical]
        assert ref['tokens_sha256'] == candidate['reference_tokens_sha256']
        rp = V.source(ref['source_path'], candidate['reference_image_sha256'])
        geometry = candidate['reference_geometry']
        photos[name] = V.encoded_photo(rp, geometry['processor_input_frame'], geometry)
        for side, grid in [('query', m['query_grid_shape']), ('reference', candidate['reference_grid_shape'])]:
            values = candidate[side + '_visibility'].numpy()
            assert values.dtype == np.float64 and np.isfinite(values).all()
            assert ((values >= 0) & (values <= 1)).all()
            assert hashlib.sha256(values.tobytes()).hexdigest() == candidate[side + '_map_sha256']
            arrays[name + '_' + side + '_visibility'] = values.reshape(grid)
        scores = candidate['old_scores']
        mass = math.sqrt(float(candidate['query_visibility'].mean()) * float(candidate['reference_visibility'].mean()))
        assert abs(mass - scores['visibility_mass']) < 1e-14
        evidence[name] = dict(physical_row=physical, reference_path=str(rp), scores=scores,
                              query_map_sha256=candidate['query_map_sha256'],
                              reference_map_sha256=candidate['reference_map_sha256'])
    case = dict(dataset='medicine', dataset_label='药盒', panel='真实纠错案例', model='COST1',
                display_id=row['original_query_id'], query_id=qid, identity=role['identity'],
                query_path=str(query_path), raw_correct=False, final_correct=True,
                raw_selected=wrong, final_selected=target, wrong_physical=wrong, target_physical=target,
                wrong_role='RAW 错误 reference', stem=stem, evidence=evidence,
                query_grid=m['query_grid_shape'], source_result=bind(FIT / 'result.json'),
                raw_payload=bind(rawfolder / 'payload.pt'), roma_payload=bind(romafolder / 'payload.pt'),
                source_query_id=source_qid, head_source=validation['payload'],
                head_parameters_hex=fitted['parameters']['COST1'], source_fold=row['fold'],
                query_excluded_from_head_training=True, decision_logit_error=error,
                selected_logit=float(sealed.max()), selection_rule=SELECTION,
                evidence_scope='H593 grouped OOF development; illustrative rescue selection',
                map_rendering='native FP64 arrays; nearest; fixed 0 to 1; no per-image normalization',
                fold_number_displayed=False, training_updates=0, encoder_or_matcher_inference_calls=0)
    np.savez_compressed(OUT / 'data' / (stem + '.npz'), **arrays)
    with np.load(OUT / 'data' / (stem + '.npz')) as loaded:
        for k, a in arrays.items():
            assert a.shape == loaded[k].shape and a.dtype == loaded[k].dtype
            assert a.tobytes() == loaded[k].tobytes()
    dump(OUT / 'data' / (stem + '.json'), case)
    return case, photos, arrays


def main():
    for folder in [OUT, OUT / 'data']:
        folder.mkdir(parents=True, exist_ok=True)
    rows, roles, workers, locations = select()
    cases = []
    with PdfPages(OUT / 'COST1_medicine_12cases.pdf') as pdf:
        for i, row in enumerate(rows, 1):
            stem = f"{i:02d}_COST1_{row['original_query_id']}"
            case, photos, arrays = extract(row, roles[row['query_id']], workers[row['query_id']], locations[row['query_id']], stem)
            fig = V.draw(case, photos, arrays, i)
            shown = '\n'.join(t.get_text() for t in fig.texts)
            assert 'fold' not in shown.lower() and '折' not in shown
            assert 'COST1' in shown and 'ORIGINAL7' not in shown
            for ext in ('png', 'pdf', 'svg'):
                fig.savefig(OUT / (stem + '.' + ext), dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
            assert Image.open(OUT / (stem + '.png')).size == (3200, 1800)
            cases.append(case)
            print('RENDERED', stem, flush=True)
    sheet = Image.new('RGB', (1920, 4 * 385 + 70), '#FAFCFE')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', 25)
    draw.text((28, 20), 'COST1  |  12 medicine retrieval rescue cases', font=font, fill=V.INK)
    for i, c in enumerate(cases):
        with Image.open(OUT / (c['stem'] + '.png')) as im:
            thumb = im.resize((640, 360), Image.Resampling.LANCZOS)
        x, y = (i % 3) * 640, (i // 3) * 385 + 70
        sheet.paste(thumb, (x, y))
    sheet.save(OUT / 'contact_sheet.jpg', quality=94)
    cards = '\n'.join(f'<article><h2>{html.escape(c["display_id"])} · COST1</h2><a href="{c["stem"]}.png"><img src="{c["stem"]}.png" loading="lazy"></a><p><a href="{c["stem"]}.png">PNG</a> · <a href="{c["stem"]}.pdf">PDF</a> · <a href="{c["stem"]}.svg">SVG</a></p></article>' for c in cases)
    (OUT / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>COST1 药盒热图</title><style>body{max-width:1400px;margin:32px auto;font:18px sans-serif;background:#fafcfe;color:#142b42}img{width:100%}a{color:#087f8c}article{margin:32px 0}</style><h1>COST1 药盒热图</h1><p><a href="COST1_medicine_12cases.pdf">合订 PDF</a> · <a href="COST1_medicine_images.zip">全部图片 ZIP</a></p>' + cards + '</html>')
    dump(OUT / 'data/cases_manifest.json', cases)
    V.source(__file__)
    V.source(V.__file__)
    dump(OUT / 'data/source_manifest.json', V.SOURCES)
    validation = dict(status='COST1_MEDICINE_VISIBILITY_RENDER_PASS', cases=len(cases),
                      distinct_identities=len({c['identity'] for c in cases}),
                      all_decisions_match_sealed_COST1_predictions=True,
                      all_queries_excluded_from_corresponding_head_training=True,
                      cached_and_exported_maps_bit_exact=True, fixed_color_scale=[0, 1],
                      fold_number_displayed=False, training_updates=0,
                      new_encoder_or_matcher_inference_calls=0,
                      maximum_logit_replay_error=max(c['decision_logit_error'] for c in cases))
    dump(OUT / 'data/validation.json', validation)
    (OUT / 'README.md').write_text('# COST1 药盒可见性图\n\n12 个已验证的成功纠错案例，来自 H593 原分组留出预测。选图只用于讲解，不代表总体准确率。图上不显示折号，模型来源及对应参数保存在 data/*.json。\n\n'+SELECTION+'\n\n所有热图直接读取已有 RoMa 缓存，保留原 token 网格和 FP64 数值，统一 0–1 色标。照片仅作显示缩放。COST1 决策已与封存输出核对，无训练、无新的编码器或匹配器前向。\n\n打开 index.html 浏览，或使用 COST1_medicine_12cases.pdf。COST1_medicine_images.zip 仅含图片与 PDF/SVG，COST1_medicine_complete.zip 另含来源数据和验证记录。\n')
    with zipfile.ZipFile(OUT / 'COST1_medicine_images.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.suffix in ('.png', '.svg', '.pdf', '.jpg'):
                z.write(p, p.name)
    with zipfile.ZipFile(OUT / 'COST1_medicine_complete.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix != '.zip':
                z.write(p, str(p.relative_to(OUT)))
    for name in ('COST1_medicine_images.zip', 'COST1_medicine_complete.zip'):
        with zipfile.ZipFile(OUT / name) as z:
            assert z.testzip() is None
            for member in z.namelist():
                assert hashlib.sha256(z.read(member)).hexdigest() == V.sha(OUT / member)
    print(json.dumps(validation), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
