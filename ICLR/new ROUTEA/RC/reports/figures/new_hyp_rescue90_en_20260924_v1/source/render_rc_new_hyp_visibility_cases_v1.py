#!/usr/bin/env python3
"""Render additional real visibility cases from frozen caches, without inference."""
from __future__ import annotations
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/figures/new_hyp_visibility_cases_20260915_v1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/new_hyp_visibility_cases_mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps, ImageDraw, ImageFont

torch.set_num_threads(2)
FONT = '/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'],
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path',
    'font.size': 12, 'xtick.color': '#586B7D', 'ytick.color': '#586B7D'})
INK, MUTED, TEAL, RED, BG = '#142B42', '#586B7D', '#087F8C', '#BA4B56', '#FAFCFE'
SOURCES, CACHE = {}, {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if key not in SOURCES:
        SOURCES[key] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    if expected is not None:
        assert SOURCES[key]['sha256'] == expected, (path, 'SOURCE_SHA')
    return path


def read(path):
    return json.loads(source(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def load_cache(folder):
    folder = Path(folder)
    if folder not in CACHE:
        receipt = read(folder / 'receipt.json')
        validation = read(folder / 'validation.json')
        assert 'PASS' in validation['status']
        path = source(receipt['payload']['path'], receipt['payload']['sha256'])
        CACHE[folder] = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    return CACHE[folder]


def collect_cases():
    work = ROOT / 'results/rc_original7_eval128_full_evidence_v1'
    med = read(work / 'result.json')
    assert read(work / 'validation.json')['status'] == 'ORIGINAL7_EVAL128_POSTJOIN_LITERAL_REPLAY_PASS'
    workers = {x['query_id']: x for x in read(med['worker_manifest']['path'])['records']}
    source(med['worker_manifest']['path'], med['worker_manifest']['sha256'])
    prejoin = {x['query_id']: x for x in read(work / 'prejoin_records.json')}
    rows = med['actions']['REAL']
    rescues = [x for x in rows if not x['base_correct'] and x['final_correct']]
    assert len(rescues) == 12
    # All medicine rescues, then two explicit contrasting examples. Selection never uses map brightness.
    selected = rescues + [next(x for x in rows if x['original_query_id'] == q)
                         for q in ['OUTCOME-0337', 'OUTCOME-0446']]
    cases = []
    for a in selected:
        assert a['candidate_recall']
        qid, ordinal = a['query_id'], a['execution_ordinal']
        target = a['target_physical_row_in_full_rank']
        wrong = a['RAW_winner_physical_row'] if not a['base_correct'] else a['final_physical_row']
        assert target != wrong
        cases.append(dict(dataset='medicine', dataset_label='药盒', panel='历史 EVAL128 内部面板',
            model='ORIGINAL7', model_lineage=med['model'], evidence_scope='历史内部结果的事后可视化',
            query_id=qid, display_id=a['original_query_id'], execution_ordinal=ordinal,
            identity=a['identity'], query_path=workers[qid]['query_image_path'],
            source_result=str(work / 'result.json'), raw_correct=a['base_correct'],
            final_correct=a['final_correct'], raw_selected=a['RAW_winner_physical_row'],
            final_selected=a['final_physical_row'], target_physical=target, wrong_physical=wrong,
            wrong_role='RAW 错误 reference' if not a['base_correct'] else '最终误选 reference',
            raw_root=str(ROOT / 'results/rc_original7_eval128_token_raw_v1'),
            roma_root=str(ROOT / 'results/rc_original7_eval128_roma_v1'),
            selection_rule='all 12 RAW-to-ORIGINAL7 rescues; plus prespecified OUTCOME-0337 and OUTCOME-0446',
            prejoin=prejoin[qid]))
    work = ROOT / 'results/rc_new_hyp_rpc_transfer_v1'
    rpc = read(work / 'result.json')
    assert 'PASS' in read(work / 'result_validation.json')['status']
    assert 'PASS' in read(work / 'independent_final_audit_v1.json')['status']
    workers = {x['query_id']: x for x in read(work / 'worker_manifest.json')['records']}
    gallery = {x['identity']: x for x in read(work / 'gallery_manifest.json')['records']}
    # The next three COST1 rescues after the already illustrated Q-0036, and the first break.
    rescues = sorted([x for x in rpc['rows'] if not x['correct']['RAW'] and x['correct']['COST1']],
                     key=lambda x: x['query_id'])
    selected = [x for x in rescues if x['query_id'] > 'RPC-Q-0036'][:3]
    selected += sorted([x for x in rpc['rows'] if x['correct']['RAW'] and not x['correct']['COST1']],
                       key=lambda x: x['query_id'])[:1]
    for a in selected:
        assert a['target_in_C128']
        qid = a['query_id']
        wrong = a['selected']['RAW'] if not a['correct']['RAW'] else a['selected']['COST1']
        cases.append(dict(dataset='rpc', dataset_label='RPC 商品', panel='RPC600 外部确认面板',
            model='COST1', model_lineage=rpc['model_lineage'], evidence_scope='外部确认结果的事后可视化',
            query_id=qid, display_id=qid, execution_ordinal=workers[qid]['execution_ordinal'],
            identity=a['identity'], query_path=workers[qid]['image_path'],
            source_result=str(work / 'result.json'), raw_correct=a['correct']['RAW'],
            final_correct=a['correct']['COST1'], raw_selected=a['selected']['RAW'],
            final_selected=a['selected']['COST1'], target_physical=gallery[a['identity']]['physical_row'],
            wrong_physical=wrong, wrong_role='RAW 错误 reference' if not a['correct']['RAW'] else '最终误选 reference',
            raw_root=str(work / 'raw'), roma_root=str(work / 'roma'),
            selection_rule='next 3 COST1 rescues after already shown RPC-Q-0036 by query_id; first COST1 break'))
    return cases


def encoded_photo(path, frame, geometry):
    with Image.open(path) as opened:
        assert list(reversed(opened.size)) == geometry['raw_size_hw']
        if frame == 'EXIF_ORIENTED_BEFORE_RESIZE':
            im = ImageOps.exif_transpose(opened).convert('RGB')
        else:
            assert frame == 'DECODED_RAW_BEFORE_EXIF'
            im = opened.convert('RGB')
    # Resizing is for the photo thumbnail only; native heatmap values are never resampled.
    im.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    return im


def extract(case, stem):
    shard = f"shard{case['execution_ordinal'] // 8:02d}"
    raw = load_cache(Path(case['raw_root']) / shard)
    roma = load_cache(Path(case['roma_root']) / shard)
    q = next(x for x in raw['records'] if x['query_id'] == case['query_id'])
    m = next(x for x in roma['records'] if x['query_id'] == case['query_id'])
    assert q['execution_ordinal'] == m['execution_ordinal'] == case['execution_ordinal']
    assert q['query_tokens_sha256'] == m['query_tokens_sha256']
    assert q['query_grid_shape'] == m['query_grid_shape']
    assert q['candidate_physical_rows'] == m['candidate_physical_rows']
    assert case['target_physical'] in q['candidate_physical_rows']
    assert case['wrong_physical'] in q['candidate_physical_rows']
    assert case['wrong_physical'] != case['target_physical']
    assert int(q['candidate_physical_rows'][q['candidate_raw_scores'].argmax()]) == case['raw_selected']
    photo_path = source(case['query_path'], m['query_source_sha256'])
    photos = {'query': encoded_photo(photo_path, m['processor_input_frame'], m['query_geometry'])}
    arrays, evidence = {}, {}
    for role in ['wrong', 'target']:
        physical = case[role + '_physical']
        c = next(x for x in m['candidates'] if x['physical_row'] == physical)
        ref = raw['references'][physical]
        assert c['reference_tokens_sha256'] == ref['tokens_sha256']
        rp = source(ref['source_path'], c['reference_image_sha256'])
        assert c['reference_grid_shape'] == ref['grid_shape']
        photos[role] = encoded_photo(rp, c['reference_geometry']['processor_input_frame'], c['reference_geometry'])
        for side, grid in [('query', m['query_grid_shape']), ('reference', c['reference_grid_shape'])]:
            a = c[side + '_visibility'].numpy()
            assert a.dtype == np.float64 and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
            assert hashlib.sha256(a.tobytes()).hexdigest() == c[side + '_map_sha256']
            arrays[role + '_' + side + '_visibility'] = a.reshape(grid)
        scores = c['old_scores']
        assert scores['physical_row'] == physical
        mass = np.sqrt(c['query_visibility'].numpy().mean() * c['reference_visibility'].numpy().mean())
        assert abs(float(mass) - scores['visibility_mass']) < 1e-14
        if 'prejoin' in case:
            for key in ['real_score', 'visibility_mass', 'query_control_score', 'reference_control_score']:
                sealed = case['prejoin']['C4_binary64'][str(c['candidate_position'])][key]
                assert scores[key].hex() == sealed, (case['query_id'], role, key)
        evidence[role] = dict(physical_row=physical, reference_path=str(rp),
            reference_grid=ref['grid_shape'], reference_geometry=c['reference_geometry'],
            scores=scores, query_map_sha256=c['query_map_sha256'], reference_map_sha256=c['reference_map_sha256'])
    case.pop('prejoin', None)
    case.update(stem=stem, query_grid=m['query_grid_shape'], query_geometry=m['query_geometry'],
        processor_input_frame=m['processor_input_frame'], evidence=evidence,
        raw_payload=str(Path(case['raw_root']) / shard / 'payload.pt'),
        roma_payload=str(Path(case['roma_root']) / shard / 'payload.pt'),
        visualization='literal cached native query token grids, fixed [0,1], nearest pixels; no overlay, no mask',
        inference_calls=0, training_updates=0)
    np.savez_compressed(OUT / 'data' / (stem + '.npz'), **arrays)
    # Verify exported arrays preserve the exact bytes and shapes used in the plot.
    with np.load(OUT / 'data' / (stem + '.npz')) as exported:
        for key, value in arrays.items():
            assert exported[key].dtype == value.dtype
            assert exported[key].shape == value.shape and exported[key].tobytes() == value.tobytes()
    for role, photo in photos.items():
        photo.save(OUT / 'assets' / f'{stem}_{role}.jpg', quality=94)
    write_json(OUT / 'data' / (stem + '.json'), case)
    return photos, arrays


def outcome(c):
    if not c['raw_correct'] and c['final_correct']:
        return '成功纠错', 'RAW 错误 → 最终正确', TEAL
    if c['raw_correct'] and not c['final_correct']:
        return '误改案例', 'RAW 正确 → 最终错误', RED
    return '尚未纠正', 'RAW 错误 → 最终仍错', RED


def draw(c, photos, arrays, number):
    kind, transition, color = outcome(c)
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(.05, .947, 'new HYP  /  ACTUAL VISIBILITY CASES', color=TEAL, fontsize=10, weight='bold')
    fig.text(.05, .887, f"{c['dataset_label']}：同一 query，面对不同 reference 的质量权重", fontsize=25, color=INK, weight='bold')
    fig.text(.05, .837, f"{c['display_id']}  ·  {kind}  ·  {c['panel']}  ·  {c['model']}", fontsize=14, color=MUTED)
    fig.add_artist(plt.Line2D([.05, .95], [.809, .809], transform=fig.transFigure, color='#DCE5ED', lw=1))
    ax = fig.add_axes([.055, .34, .26, .415]); ax.imshow(photos['query']); ax.axis('off')
    ax.set_title('同一 Query（完整照片）', fontsize=14, color=INK, pad=12)
    for j, role in enumerate(['wrong', 'target']):
        x = .385 + j * .30
        ax = fig.add_axes([x, .34, .235, .415])
        values = arrays[role + '_query_visibility']
        im = ax.imshow(values, cmap='viridis', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
        assert im.get_array().tobytes() == values.tobytes()
        ax.set_title('与' + (c['wrong_role'] if role == 'wrong' else '正确 reference') + '配对', fontsize=13, pad=12)
        ax.set_xlabel('query token 列', fontsize=10); ax.set_ylabel('query token 行', fontsize=10)
        ax.tick_params(labelsize=9, length=3)
        for spine in ax.spines.values(): spine.set_visible(False)
        sc = c['evidence'][role]['scores']
        fig.text(x, .273, f"S = {sc['real_score']:.5f}    M = {sc['visibility_mass']:.5f}", color=INK, fontsize=13)
    cbax = fig.add_axes([.405, .218, .485, .015])
    cb = fig.colorbar(im, cax=cbax, orientation='horizontal', ticks=[0, .5, 1])
    cb.outline.set_edgecolor('#DCE5ED'); cb.ax.tick_params(labelsize=9, length=3)
    fig.text(.055, .281, transition, fontsize=15, color=color, weight='bold')
    fig.text(.055, .238, f"候选：RAW 自然 C128\n动作：{c['model']} HOLD / SWITCH", fontsize=11, color=MUTED, linespacing=1.6, va='top')
    fig.text(.055, .145, 'S：加权内容匹配分数\nM：query/reference 联合可见性\n最终决策联合使用多项证据。', fontsize=10, color=MUTED, linespacing=1.55, va='top')
    for j, role in enumerate(['wrong', 'target']):
        x = .385 + j * .30
        ax = fig.add_axes([x, .071, .093, .111]); ax.imshow(photos[role]); ax.axis('off')
        label = c['wrong_role'] if role == 'wrong' else '正确 reference'
        fig.text(x + .105, .155, label, fontsize=10, color=INK)
        fig.text(x + .105, .126, f"图库条目 {c[role + '_physical']}", fontsize=10, color=MUTED)
        fig.text(x + .105, .098, '实际 reference 缩略图', fontsize=9, color=MUTED)
    fig.text(.05, .027, '实际缓存 · 所有热图固定 0–1 色标 · token 权重不是分割 / ownership · 个案用于讲解，不代表整体成功率', fontsize=9, color=MUTED)
    fig.text(.95, .027, f'{number:02d}', ha='right', color=TEAL, fontsize=10)
    return fig


def package(cases):
    fields = ['dataset', 'panel', 'model', 'display_id', 'query_id', 'identity', 'raw_correct', 'final_correct',
              'raw_selected', 'final_selected', 'target_physical', 'wrong_physical', 'wrong_role']
    table = []
    for c in cases:
        row = {k: c[k] for k in fields}
        for role in ['wrong', 'target']:
            sc = c['evidence'][role]['scores']
            row[role + '_S'] = sc['real_score']; row[role + '_M'] = sc['visibility_mass']
        row.update(png=c['stem'] + '.png', source_result=c['source_result'], roma_payload=c['roma_payload'])
        table.append(row)
    with (OUT / 'data/cases.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    write_json(OUT / 'cases_manifest.json', cases)
    source(__file__)
    write_json(OUT / 'source_manifest.json', SOURCES)
    font = ImageFont.truetype(FONT, 23)
    latin_font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), 23)
    sheet = Image.new('RGB', (1920, 6 * 405 + 80), '#FAFCFE'); d = ImageDraw.Draw(sheet)
    def sheet_text(x, y, text):
        # DroidSansFallback covers Chinese but not Latin; PIL has no automatic fallback.
        for char in text:
            selected_font = font if ord(char) > 255 else latin_font
            d.text((x, y), char, font=selected_font, fill=INK)
            x += selected_font.getlength(char)
    sheet_text(30, 22, '18 个真实配对案例：14 张药盒 + 4 张 RPC 商品')
    cards = []
    for i, c in enumerate(cases):
        with Image.open(OUT / (c['stem'] + '.png')) as im:
            assert im.size == (3200, 1800)
            thumb = im.resize((640, 360), Image.Resampling.LANCZOS)
        x, y = (i % 3) * 640, (i // 3) * 405 + 80
        sheet.paste(thumb, (x, y))
        sheet_text(x + 18, y + 365, f"{i+1:02d} {c['display_id']} · {outcome(c)[0]}")
        st = c['stem']; label = html.escape(c['display_id'] + ' · ' + outcome(c)[0] + ' · ' + c['model'])
        cards.append(f'<article><h2>{label}</h2><a href="{st}.png"><img src="{st}.png" loading="lazy" alt="{label}"></a>'
            f'<p><a href="{st}.png">高清 PNG</a> · <a href="{st}.pdf">PDF</a> · <a href="{st}.svg">SVG</a>'
            f' · 原照片：<a href="assets/{st}_query.jpg">query</a> / <a href="assets/{st}_wrong.jpg">错误 reference</a>'
            f' / <a href="assets/{st}_target.jpg">正确 reference</a> · <a href="data/{st}.json">数值与来源</a></p></article>')
    sheet.save(OUT / 'contact_sheet.jpg', quality=93)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>new HYP：真实配对可见性图册</title><style>body{margin:0;background:#FAFCFE;color:#142B42;font:17px/1.6 system-ui,sans-serif}main{max-width:1300px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:20px}a{color:#087F8C}article{margin:30px 0;padding:16px;border:1px solid #DCE5ED;border-radius:10px}img{width:100%;height:auto}p{color:#586B7D}</style><main>
<h1>同一 query，不同 reference：18 个真实案例</h1>
<p>14 张药盒来自历史 EVAL128 的 ORIGINAL7：全部 12 次纠错、1 次误改和 1 次未纠正。另补 4 张 RPC600 的固定 full-H593 COST1 案例：3 次纠错、1 次误改。两套模型和面板分别标明。</p>
<p>热图直接读取已验证缓存，保留原 token 网格，统一使用 0–1 色标；reference 缩略图为实际输入。S 为加权内容分数，M = sqrt(mean(wq) × mean(wr))。最终动作还使用其他证据，不能凭热图亮度或 S 单项代替模型决策。</p>
<p>选图为事后讲解，不估计成功率。网格不是分割，也不是 ownership 证据；本次没有训练或模型推理。</p>
<p><a href="visibility_cases_18pages.pdf">下载完整 18 页 PDF</a> · <a href="data/cases.csv">统计表 CSV</a> · <a href="contact_sheet.jpg">总览图</a></p>
''' + '\n'.join(cards) + '</main></html>'
    (OUT / 'index.html').write_text(page)
    (OUT / 'README.md').write_text('''# 真实 reference 配对可见性图册

共 18 张 3200×1800 高清图（PNG / SVG / 单页 PDF），以及 18 页合订 PDF、离线 HTML、总览 JPG、逐例 CSV / JSON / NPZ。

- 药盒：历史 ORIGINAL7 / EVAL128 / RAW 自然 C128 / HOLD-SWITCH。包含全部 12 次 RAW→ORIGINAL7 纠错；另展示 OUTCOME-0337 误改、OUTCOME-0446 未纠正。原总体成绩 RAW 88/128→99/128 不变。
- RPC：固定 full-H593 COST1 / RPC600 外部确认 / RAW 自然 C128 / HOLD-SWITCH。选择原已展示 RPC-Q-0036 之后按 query_id 排序的前三个纠错和第一个误改。原总体成绩 RAW 192/600→207/600 不变。
- 此处“错误 reference”取 RAW 错误首选；误改案例取最终错误首选，图中显式改为“最终误选 reference”。正确 reference 来自原评测身份连接，且在原 C128 内。
- 图内 S 是候选的实际加权 MaxSim 分数，M 是 sqrt(mean(wq) × mean(wr))，都不是最终头的动作分数。最终决策使用多项证据。
- 热图使用缓存的 FP64 数组，native grid + nearest 像素 + 固定 [0,1]，没有逐例归一化。照片保持完整画面和编码坐标框架，只作显示缩放。没有绘制物体 mask。
- 图片、缓存、原结果、程序哈希见 source_manifest.json。导出的 NPZ 与缓存逐字节核对；药盒 S/M 等 C4 标量还与封存 prejoin 的 FP64 hex 值核对。
- 事后案例选择用于讲解，不代表总体成功率或新增科学试验；不训练、不推理、不调整模型、阈值或候选。ownership 保留为未来方向。

打开 index.html 浏览；visibility_cases_18pages.pdf 可直接展示；assets 保存各输入的便携缩略图。
复现：OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-colpali/bin/python 'ICLR/new ROUTEA/RC/programs/render_rc_new_hyp_visibility_cases_v1.py'（从 benchmark 根目录执行）。
''')
    validation = dict(status='REAL_VISIBILITY_CASES_RENDERED_AND_DATA_VERIFIED', cases=len(cases),
        medicine_cases=sum(c['dataset'] == 'medicine' for c in cases), rpc_cases=sum(c['dataset'] == 'rpc' for c in cases),
        maps=len(cases)*2, reference_maps_exported=len(cases)*2, fixed_color_limits=[0, 1],
        source_bindings=len(SOURCES), cached_tensor_sha_verified=True, exported_npz_bytes_verified=True,
        medicine_scores_match_prejoin_binary64=True, query_and_reference_image_bytes_verified=True,
        raw_winner_matches_cache=True, source_CPU_validation_pass=True, photo_frame_matches_cache=True,
        training_updates=0, inference_calls=0, modified_experimental_artifacts=0,
        original_showcase_unchanged=True, ownership_claimed=False)
    write_json(OUT / 'validation.json', validation)
    archive = OUT / 'new_HYP_visibility_18cases.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p != archive and p.suffix != '.log':
                z.write(p, p.relative_to(OUT))
    print(json.dumps(validation, ensure_ascii=False), flush=True)
    print('PACKAGE', archive, archive.stat().st_size, sha(archive), flush=True)


def main():
    for p in [OUT, OUT / 'assets', OUT / 'data']:
        p.mkdir(parents=True, exist_ok=True)
    cases = collect_cases()
    assert len(cases) == 18
    with PdfPages(OUT / 'visibility_cases_18pages.pdf') as pdf:
        for i, c in enumerate(cases, 1):
            stem = f"{i:02d}_{c['dataset']}_{c['display_id']}"
            photos, arrays = extract(c, stem)
            fig = draw(c, photos, arrays, i)
            fig.savefig(OUT / (stem + '.png'), dpi=200, facecolor=BG)
            fig.savefig(OUT / (stem + '.pdf'), facecolor=BG)
            fig.savefig(OUT / (stem + '.svg'), facecolor=BG)
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
            print('RENDERED', i, c['display_id'], outcome(c)[0], flush=True)
    package(cases)


if __name__ == '__main__':
    main()
