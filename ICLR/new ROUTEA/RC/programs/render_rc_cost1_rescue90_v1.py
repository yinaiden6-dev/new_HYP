#!/usr/bin/env python3
"""Ninety additional COST1 rescue figures, from sealed decisions and native caches."""
from __future__ import annotations
import collections
import copy
import csv
import functools
import hashlib
import html
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rc_rescue90_mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')
import render_rc_new_hyp_visibility_cases_v1 as V
import render_rc_cost1_medicine_visibility_v1 as M
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/figures/new_hyp_rescue90_20260924_v1'
OLD18 = ROOT / 'reports/figures/new_hyp_visibility_cases_20260915_v1'
DATA = ROOT / 'results/rc_new_hyp593_oof5_v1'
FIT = ROOT / 'results/rc_six_cause_isolation_v1/loss_binding'
SELECTION = ('Exclude medicine queries in the original18; include all remaining DIFFICULT and NDV2 rescues; '
             'then fill to30 by original_query_id, prioritizing identities not yet selected. '
             'No visibility values or map brightness used for selection.')


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def binding(p):
    p = V.source(p)
    return dict(path=str(p), sha256=V.sha(p))


def medicine_selection():
    result = V.read(FIT / 'result.json')
    validation = V.read(FIT / 'validation.json')
    assert validation['result'] == binding(FIT / 'result.json')
    assert validation['status'] == 'SIX_CAUSE_LOSS_BINDING_PARITY_COUNTS_PASS'
    roles = {r['query_id']: r for r in V.read(DATA / 'metadata/curator_roles.json')['records']}
    workers = V.read(DATA / 'metadata/worker_manifest.json')['records']
    locations = {}
    for lane in ('reuse', 'missing'):
        group = [w for w in workers if ('reuse' in w) == (lane == 'reuse')]
        for i, worker in enumerate(group):
            locations[worker['query_id']] = (lane, i // 8)
    workers = {w['query_id']: w for w in workers}
    old = V.read(OLD18 / 'cases_manifest.json')
    exclude = {c['display_id'] for c in old if c['dataset'] == 'medicine'}
    rescues = [r for r in result['rows'] if not r['correct']['RAW'] and r['correct']['COST1']]
    assert len(rescues) == 58
    eligible = sorted([r for r in rescues if r['original_query_id'] not in exclude],
                      key=lambda r: r['original_query_id'])
    selected = [r for r in eligible if r['original_query_id'].startswith(('DIFFICULT-', 'NDV2-'))]
    seen = {roles[r['query_id']]['identity'] for r in selected}
    for r in eligible:
        if len(selected) == 30:
            break
        ident = roles[r['query_id']]['identity']
        if ident not in seen:
            selected.append(r)
            seen.add(ident)
    for r in eligible:
        if len(selected) == 30:
            break
        if r not in selected:
            selected.append(r)
    assert len(selected) == 30 and len({r['query_id'] for r in selected}) == 30
    dump(OUT / 'data/medicine_selection.json', dict(selection_rule=SELECTION, selected=selected,
         eligible_count=len(eligible), total_COST1_rescues=58, excluded_old18_display_ids=sorted(exclude),
         difficult_count=sum(r['original_query_id'].startswith('DIFFICULT-') for r in selected),
         new_difficult_count=sum(r['original_query_id'].startswith('NDV2-') for r in selected),
         distinct_identities=len(seen), source_result=binding(FIT / 'result.json')))
    return selected, roles, workers, locations


def augment_medicine_decision(case, row):
    fitted = V.read(case['head_source']['path'])
    pred = next(x for x in fitted['predictions'] if x['query_id'] == case['query_id'])['models']['COST1']
    payload = M.cache(Path(case['raw_payload']['path']).parent)
    q = next(x for x in payload['records'] if x['query_id'] == case['source_query_id'])
    case['sealed_COST1_logits_hex'] = pred['logits_hex']
    case['challenger_physical_rows'] = [x for x in q['candidate_physical_rows'] if x != case['raw_selected']]
    case['panel'] = 'H593 药盒面板'
    case['selection_rule'] = SELECTION
    case['track'] = row['original_query_id'].split('-')[0]
    roma = M.cache(Path(case['roma_payload']['path']).parent)
    cached = next(x for x in roma['records'] if x['query_id'] == case['source_query_id'])
    case['query_geometry'] = cached['query_geometry']
    case['processor_input_frame'] = cached['processor_input_frame']


def verify_action(case):
    z = np.array([float.fromhex(v) for v in case['sealed_COST1_logits_hex']], dtype=np.float64)
    challengers = case['challenger_physical_rows']
    assert len(z) == len(challengers) == len(set(challengers)) == 127
    assert case['raw_selected'] not in challengers and np.isfinite(z).all()
    selected = challengers[int(z.argmax())] if z.max() > 0 else case['raw_selected']
    assert selected == case['final_selected'] == case['target_physical']
    assert case['wrong_physical'] == case['raw_selected'] != selected
    assert case['raw_correct'] is False and case['final_correct'] is True
    case['selected_logit'] = float(z.max())
    case['raw_hold_utility'] = 0.0
    case['scores128_physical_axis'] = [case['raw_selected']] + challengers
    case['scores128_hex'] = [0.0.hex()] + case['sealed_COST1_logits_hex']
    case['action'] = 'SWITCH'
    case['sealed_action_independently_reselected'] = True


def prepare_dirs(group):
    folder = OUT / group
    for sub in ('', 'assets', 'data'):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    V.OUT = M.OUT = folder
    return folder


def finish_case(case, photos, arrays, folder, number, overall_pdf, group_pdf):
    verify_action(case)
    case['number'] = number
    case['relative_stem'] = f"{case['dataset']}/{case['stem']}"
    case['fold_number_displayed'] = False
    case['training_updates'] = case['inference_calls'] = 0
    for role, photo in photos.items():
        photo.save(folder / 'assets' / f"{case['stem']}_{role}.jpg", quality=95)
    path = folder / 'data' / (case['stem'] + '.npz')
    np.savez_compressed(path, **arrays)
    with np.load(path) as saved:
        for key, array in arrays.items():
            assert saved[key].dtype == np.float64
            assert saved[key].shape == array.shape and saved[key].tobytes() == array.tobytes()
    dump(folder / 'data' / (case['stem'] + '.json'), case)
    fig = V.draw(case, photos, arrays, number)
    if case['dataset'] == 'grozi':
        fig.axes[0].set_title('同一 Query（数据集商品裁图）', fontsize=14, color=V.INK, pad=12)
    texts = '\n'.join(t.get_text() for t in fig.texts)
    assert 'fold' not in texts.lower() and '折' not in texts and 'COST1' in texts
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(folder / (case['stem'] + '.' + ext), dpi=200)
    overall_pdf.savefig(fig)
    group_pdf.savefig(fig)
    V.plt.close(fig)
    with Image.open(folder / (case['stem'] + '.png')) as im:
        assert im.size == (3200, 1800)
    print(json.dumps(dict(stage='rendered', number=number, dataset=case['dataset'],
                         query=case['display_id'], png=case['relative_stem']+'.png'),ensure_ascii=False),flush=True)
    return case


def contact_sheet(cases, group):
    sheet = Image.new('RGB', (1920, 10 * 390 + 70), V.BG)
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(V.font_manager.findfont('DejaVu Sans'), 26)
    d.text((25, 20), f'{group.upper()} | 30 COST1 rescue cases', font=font, fill=V.INK)
    for i, c in enumerate(cases):
        with Image.open(OUT / (c['relative_stem']+'.png')) as im:
            thumb=im.resize((640,360),Image.Resampling.LANCZOS)
        x,y=(i%3)*640,(i//3)*390+70
        sheet.paste(thumb,(x,y))
        d.text((x+15,y+363),c['display_id'],font=font,fill=V.INK)
    sheet.save(OUT / group / 'contact_sheet.jpg',quality=93)


def package(cases, old_bindings):
    assert len(cases)==90
    counts=collections.Counter(c['dataset'] for c in cases)
    assert counts==dict(medicine=30,grozi=30,isic=30)
    assert len({(c['dataset'],c['query_id']) for c in cases})==90
    dump(OUT/'cases_manifest.json',cases)
    rows=[]
    for c in cases:
        row={k:c[k] for k in ('number','dataset','model','display_id','query_id','identity','raw_selected','final_selected','selected_logit')}
        row.update(raw_correct=False,final_correct=True,png=c['relative_stem']+'.png',source_fold=c.get('source_fold',''))
        for role in ('wrong','target'):
            row[role+'_S']=c['evidence'][role]['scores']['real_score']
            row[role+'_M']=c['evidence'][role]['scores']['visibility_mass']
        rows.append(row)
    with (OUT/'data/cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for group in counts:
        contact_sheet([c for c in cases if c['dataset']==group],group)
    cards=[]
    for c in cases:
        st=c['relative_stem'];name=html.escape(c['display_id']);ds=c['dataset']
        cards.append(f'<article data-dataset="{ds}"><h2>{c["number"]:02d} · {name} · COST1</h2>'
                     f'<a href="{st}.png"><img src="{st}.png" loading="lazy" alt="{name}"></a>'
                     f'<p><a href="{st}.png">PNG</a> · <a href="{st}.pdf">PDF</a> · <a href="{st}.svg">SVG</a>'
                     f' · <a href="{ds}/data/{c["stem"]}.json">完整决策与来源</a></p></article>')
    header='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>new HYP · 90个成功纠错案例</title><style>body{max-width:1400px;margin:30px auto;padding:0 20px;font:17px/1.6 system-ui;background:#fafcfe;color:#142b42}a{color:#087f8c}article{margin:28px 0;border:1px solid #dce5ed;border-radius:10px;padding:16px}img{width:100%;height:auto}button{margin:8px;padding:8px 18px;font:inherit}article[hidden]{display:none}</style>
<h1>90个COST1真实成功纠错案例</h1><p>药盒30 · GroZi30 · ISIC30。全部为RAW错误、COST1纠正；选图用于展示，不估计成功率。原18张图保持不变。</p>
<p>药盒：H593原分组留出头；GroZi/ISIC：固定full-H593 COST1头、无外部训练。ISIC用于同一皮肤病灶的检索，属于探索面板。</p>
<p>所有热图使用实际缓存与统一0–1色标。S、M是模型的输入证据；最终动作使用七参数头，不能用单个热图亮度代替决策。</p>
<p><a href="rescue90.pdf">90页PDF</a> · <a href="medicine/medicine_30cases.pdf">药盒PDF</a> · <a href="grozi/grozi_30cases.pdf">GroZi PDF</a> · <a href="isic/isic_30cases.pdf">ISIC PDF</a> · <a href="data/cases.csv">统计表</a> · <a href="README.md">来源说明</a></p>
<nav><button onclick="show('all')">全部90张</button><button onclick="show('medicine')">药盒30张</button><button onclick="show('grozi')">GroZi30张</button><button onclick="show('isic')">ISIC30张</button></nav>
'''
    (OUT/'index.html').write_text(header+'\n'.join(cards)+"<script>function show(d){document.querySelectorAll('article').forEach(x=>x.hidden=d!=='all'&&x.dataset.dataset!==d)}</script></html>")
    (OUT/'README.md').write_text('''# COST1新增90张成功纠错图

药盒30张、GroZi30张、ISIC30张；每例PNG 3200×1800、SVG、单页PDF。另有90页总PDF、各30页PDF、各组总览图、离线index.html、JSON/CSV/原生NPZ与照片缩略图。

- 药盒：H593 COST1原分组留出预测，共58救回/3误改（RAW426→481/593）。排除原18图已有药盒query后，有48个成功例；纳入全部12个DIFFICULT、1个NDV2，再补17个OUTCOME。30例覆盖24个身份。图上不显示折号；data保留对应fold、头参数、训练排除和完整127动作分数。旧18图的药盒头是ORIGINAL7，本组明确使用COST1。
- GroZi：固定full-H593 COST1，RAW321→355/480，共34救回/0误改；选30例，优先覆盖不同商品身份。query为原实验使用的数据集商品裁图，不是未裁剪全场景。自然C128来自原混合图库，错误reference可能来自历史药盒图库，不能替换成更好看的商品图。
- ISIC：固定full-H593 COST1，RAW466→500/537，共34救回/0误改；排除旧展示ISIC-Q-0013后选30，优先覆盖不同病灶。属于已打开的探索面板，不是新外部确认，也不是疾病诊断。三张原图的ISIC ID、已有署名和许可证保留在data/isic_candidates.json及逐例JSON；不将整个ISIC库作统一许可推断。

图的布局沿用原18张：同一完整query、与RAW错误reference及正确reference配对的query可见性网格，附真实reference缩略图、S和M。网格保留缓存FP64原值与原尺寸，nearest显示、固定0–1，无逐图归一化。它是匹配可见性权重，不是分割或ownership证明。M=sqrt(mean(wq)×mean(wr))；S为原加权内容分数，都不是最终动作logit。

完整候选决策：data中保留127个封存挑战者logit及RAW的HOLD=0，物理reference轴可复核。本次只选择和绘制既有成功案例，无训练、无编码器或RoMa前向，无新增准确率结论；没有改动原18图及实验结果。旧18图位于../new_hyp_visibility_cases_20260915_v1。

复现程序：RC/programs/render_rc_cost1_rescue90_v1.py。所有来源、数组、选择和校验记录随完整ZIP保存。本包为本地展示材料，本轮未上传GitHub。
''')
    for p,b in old_bindings.items():
        assert V.sha(Path(p))==b,'ORIGINAL18_CHANGED'
    V.source(__file__);V.source(V.__file__);V.source(M.__file__)
    dump(OUT/'source_manifest.json',V.SOURCES)
    validation=dict(status='COST1_RESCUE90_RENDER_AND_SEALED_DECISION_PASS',cases=90,datasets=dict(counts),
                    all_raw_wrong_final_correct=True,full127_logits_retained_and_action_reselected=True,
                    native_FP64_maps_bit_exact=True,fixed_color_scale=[0,1],query_heatmaps=180,
                    reference_maps_also_exported=180,fold_number_displayed=False,original18_unchanged=True,
                    old18_sha256=old_bindings,inference_calls=0,training_updates=0)
    dump(OUT/'validation.json',validation)
    archive=OUT/'new_HYP_rescue90_complete.zip'
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ('.zip','.log','.out','.err','.partial')]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read(str(p.relative_to(OUT)))).hexdigest()==V.sha(p)
    dump(OUT/'archive_validation.json',dict(status='ZIP_LOOSE_FILES_BYTE_IDENTICAL',files=len(files),
          archive=str(archive),bytes=archive.stat().st_size,sha256=V.sha(archive)))
    print(json.dumps(validation),flush=True)
    print('ARCHIVE',archive,archive.stat().st_size,flush=True)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    original_paths=[p for p in sorted(OLD18.rglob('*')) if p.is_file()]
    old_bindings={str(p):V.sha(p) for p in original_paths}
    # Bound the number of mmap-backed shard objects; never invoke models.
    M.cache=functools.lru_cache(maxsize=4)(M.cache)
    V.load_cache=M.cache
    M.SELECTION=SELECTION
    selected,roles,workers,locations=medicine_selection()
    external={g:V.read(OUT/'data'/f'{g}_candidates.json') for g in ('grozi','isic')}
    assert all(len(v['selected'])==30 for v in external.values())
    cases=[]
    with PdfPages(OUT/'rescue90.pdf') as pdf:
        for group in ('medicine','grozi','isic'):
            folder=prepare_dirs(group)
            inputs=selected if group=='medicine' else external[group]['selected']
            with PdfPages(folder/f'{group}_30cases.pdf') as group_pdf:
                for row in inputs:
                    number=len(cases)+1
                    display=row['original_query_id'] if group=='medicine' else row['display_id']
                    stem=f'{number:02d}_{group}_{display}'
                    if group=='medicine':
                        qid=row['query_id']
                        case,photos,arrays=M.extract(row,roles[qid],workers[qid],locations[qid],stem)
                        augment_medicine_decision(case,row)
                    else:
                        case=copy.deepcopy(row)
                        if 'sealed_COST1_logits_hex' not in case:
                            payload=V.read(case['prediction_payload'])
                            pr=next(r for r in payload['records'] if r['query_id']==case['query_id'])
                            case['sealed_COST1_logits_hex']=pr['models']['COST1']['logits_hex']
                            case['challenger_physical_rows']=[p for p in sorted(pr['raw_ranked_physical_rows'][:128]) if p!=pr['raw_selected']]
                        photos,arrays=V.extract(case,stem)
                    cases.append(finish_case(case,photos,arrays,folder,number,pdf,group_pdf))
    package(cases,old_bindings)


if __name__=='__main__':
    main()
