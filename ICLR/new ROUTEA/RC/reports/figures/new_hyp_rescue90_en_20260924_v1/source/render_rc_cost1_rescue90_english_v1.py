#!/usr/bin/env python3
"""English-only figures and booklet, reusing the verified rescue90 arrays and photos."""
from __future__ import annotations
import copy
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rc_rescue90_english_mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')
import render_rc_new_hyp_visibility_cases_v1 as V
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/figures/new_hyp_rescue90_20260924_v1'
OUT=ROOT/'reports/figures/new_hyp_rescue90_en_20260924_v1'
LABELS={'medicine':'Medicine packaging','grozi':'GroZi products','isic':'ISIC lesions'}
PANELS={'medicine':'H593 development panel','grozi':'GroZi480 external panel','isic':'ISIC537 exploratory panel'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def bind(p):return dict(path=str(p),sha256=sha(p))


def draw(case,photos,arrays):
    fig=V.draw(case,photos,arrays,case['number'])
    assert len(fig.texts)==16 and len(fig.axes)==6
    s1=case['evidence']['wrong']['scores'];s2=case['evidence']['target']['scores']
    texts=[
      'new HYP  /  REFERENCE-CONDITIONED VISIBILITY',
      'How the reference changes query visibility',
      f"{case['display_id']}  |  {PANELS[case['dataset']]}  |  COST1  |  Successful correction",
      f"S = {s1['real_score']:.5f}    M = {s1['visibility_mass']:.5f}",
      f"S = {s2['real_score']:.5f}    M = {s2['visibility_mass']:.5f}",
      'RAW incorrect → COST1 correct',
      'Candidates: natural RAW C128\nDecision: COST1 HOLD / SWITCH',
      'S: weighted content score\nM: joint visibility mass\nThe head combines multiple signals.',
      'RAW incorrect reference',f"Gallery entry {case['wrong_physical']}",'Actual reference thumbnail',
      'Correct reference',f"Gallery entry {case['target_physical']}",'Actual reference thumbnail',
      'Actual cached token weights  |  Shared 0–1 scale  |  Not a segmentation mask  |  Selected rescues, not an accuracy estimate',
      f"{case['number']:02d}",
    ]
    for obj,text in zip(fig.texts,texts):obj.set_text(text)
    fig.texts[1].set_fontsize(26)
    fig.texts[2].set_fontsize(12.5)
    fig.texts[5].set_fontsize(12)
    fig.texts[6].set_fontsize(9.5)
    fig.texts[7].set_fontsize(9.5)
    fig.texts[14].set_fontsize(8)
    fig.axes[0].set_title('Query (dataset product crop)' if case['dataset']=='grozi'
                         else 'Query (complete input image)',fontsize=13,color=V.INK,pad=12)
    for ax,title in zip(fig.axes[1:3],["Paired with RAW's incorrect reference",'Paired with the correct reference']):
        ax.set_title(title,fontsize=12,pad=12)
        ax.set_xlabel('Query token column',fontsize=10)
        ax.set_ylabel('Query token row',fontsize=10)
    # Give each reference a full-column panel instead of the original tiny thumbnail.
    # Native arrays and image aspect ratios remain unchanged.
    for j in range(2):
        x=.385+j*.30
        fig.axes[1+j].set_position([x,.455,.235,.30])
        fig.texts[3+j].set_position((x,.383))
        fig.axes[4+j].set_position([x,.05,.235,.235])
        role_index=8+3*j
        physical=case['wrong_physical'] if j==0 else case['target_physical']
        label='Incorrect reference (RAW)' if j==0 else 'Correct reference'
        fig.texts[role_index].set_text(f'{label} · entry {physical}')
        fig.texts[role_index].set_position((x+.1175,.303))
        fig.texts[role_index].set_ha('center')
        fig.texts[role_index].set_fontsize(10.5)
        fig.texts[role_index+1].set_visible(False)
        fig.texts[role_index+2].set_visible(False)
    fig.axes[3].set_position([.405,.344,.485,.012])
    shown=[t.get_text() for t in fig.texts if t.get_visible()]+[a.get_title() for a in fig.axes]+[a.get_xlabel() for a in fig.axes]+[a.get_ylabel() for a in fig.axes]
    assert not any(re.search('[\u3400-\u9fff]',s) for s in shown)
    return fig,shown


def translate_case(original):
    c=copy.deepcopy(original);g=c['dataset']
    c.update(dataset_label=LABELS[g],panel=PANELS[g],wrong_role='RAW incorrect reference',
             language='English',evidence_scope=('Grouped held-out development predictions' if g=='medicine'
             else 'Post-hoc illustrations from frozen external predictions'),
             visualization='Native cached FP64 grids; nearest pixels; fixed color scale [0,1]; no overlay or mask',
             source_case=bind(BASE/g/'data'/(c['stem']+'.json')))
    return c


def contact(cases,group):
    canvas=Image.new('RGB',(1920,3970),V.BG);d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(V.font_manager.findfont('DejaVu Sans'),26)
    d.text((25,20),f'{LABELS[group]} | 30 successful COST1 corrections',font=font,fill=V.INK)
    for i,c in enumerate(cases):
        with Image.open(OUT/(c['relative_stem']+'.png')) as im:thumb=im.resize((640,360),Image.Resampling.LANCZOS)
        x,y=(i%3)*640,(i//3)*390+70;canvas.paste(thumb,(x,y))
        d.text((x+15,y+363),c['display_id'],font=font,fill=V.INK)
    canvas.save(OUT/group/'contact_sheet.jpg',quality=93)


def main():
    torch.set_num_threads(2)
    base_validation=read(BASE/'validation.json')
    independent=read(BASE/'independent_validation.json')
    assert 'PASS' in base_validation['status'] and 'PASS' in independent['status']
    originals=read(BASE/'cases_manifest.json');assert len(originals)==90
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['data','source']:
        shutil.copytree(BASE/name,OUT/name,dirs_exist_ok=True)
    # Keep original provenance text and licenses verbatim; only the presentation is translated.
    shutil.copy2(BASE/'independent_validation.json',OUT/'data/source_independent_validation.json')
    shutil.copy2(BASE/'source_manifest.json',OUT/'data/source_cache_manifest.json')
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    shutil.copy2(ROOT/'programs/verify_rc_cost1_rescue90_independent_v1.py',OUT/'source/verify_rc_cost1_rescue90_independent_v1.py')
    cases=[];captions={}
    with PdfPages(OUT/'rescue90.pdf') as pdf:
        for group in ['medicine','grozi','isic']:
            folder=OUT/group
            for sub in ['assets','data']:(folder/sub).mkdir(parents=True,exist_ok=True)
            with PdfPages(folder/f'{group}_30cases.pdf') as group_pdf:
                for original in [c for c in originals if c['dataset']==group]:
                    c=translate_case(original);stem=c['stem'];source=BASE/group
                    c['layout']='English; full-column reference panels; reference boxes .235 by .235 of the canvas'
                    npz=source/'data'/(stem+'.npz')
                    shutil.copy2(npz,folder/'data'/npz.name)
                    assert sha(npz)==sha(folder/'data'/npz.name)
                    with np.load(npz) as loaded:arrays={k:loaded[k].copy() for k in loaded.files}
                    for role in ['wrong','target']:
                        for side in ['query','reference']:
                            a=arrays[f'{role}_{side}_visibility']
                            assert a.dtype==np.float64 and hashlib.sha256(a.tobytes()).hexdigest()==c['evidence'][role][side+'_map_sha256']
                    photos={}
                    for role in ['query','wrong','target']:
                        p=source/'assets'/f'{stem}_{role}.jpg';dest=folder/'assets'/p.name
                        shutil.copy2(p,dest)
                        with Image.open(p) as im:photos[role]=im.copy()
                    fig,shown=draw(c,photos,arrays)
                    for ext in ['png','pdf','svg']:fig.savefig(folder/(stem+'.'+ext),dpi=200)
                    pdf.savefig(fig);group_pdf.savefig(fig);V.plt.close(fig)
                    with Image.open(folder/(stem+'.png')) as im:assert im.size==(3200,1800)
                    dump(folder/'data'/(stem+'.json'),c)
                    cases.append(c);captions[stem]=shown
                    print(json.dumps(dict(stage='english_rendered',number=c['number'],dataset=group,query=c['display_id'])),flush=True)
    dump(OUT/'cases_manifest.json',cases);dump(OUT/'data/rendered_english_text.json',captions)
    shutil.copy2(BASE/'data/cases.csv',OUT/'data/cases.csv')
    for group in LABELS:contact([c for c in cases if c['dataset']==group],group)
    cards=[]
    for c in cases:
        s=c['relative_stem'];g=c['dataset'];label=html.escape(c['display_id'])
        cards.append(f'<article data-dataset="{g}"><h2>{c["number"]:02d} · {label} · COST1</h2><a href="{s}.png"><img src="{s}.png" loading="lazy" alt="{label}"></a><p><a href="{s}.png">PNG</a> · <a href="{s}.pdf">PDF</a> · <a href="{s}.svg">SVG</a> · <a href="{g}/data/{c["stem"]}.json">Decision data and provenance</a></p></article>')
    header='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>new HYP: 90 successful corrections</title>
<style>body{max-width:1400px;margin:30px auto;padding:0 20px;font:17px/1.6 system-ui;background:#fafcfe;color:#142b42}a{color:#087f8c}article{margin:28px 0;border:1px solid #dce5ed;border-radius:10px;padding:16px}img{width:100%;height:auto}button{margin:8px;padding:8px 18px;font:inherit}article[hidden]{display:none}</style>
<h1>90 successful COST1 corrections</h1><p>30 medicine queries · 30 GroZi queries · 30 ISIC queries. All cases change an incorrect RAW prediction into a correct COST1 prediction under the sealed identity labels.</p>
<p>These are selected illustrations, not an accuracy estimate. Medicine uses the original H593 grouped held-out heads; GroZi and ISIC use the frozen full-H593 COST1 head. ISIC is an exploratory same-lesion retrieval panel.</p>
<p>Heatmaps use actual cached values and a shared 0–1 color scale. S is weighted content similarity; M is joint visibility mass. The seven-parameter head combines multiple signals. Heatmap brightness alone does not determine the final action.</p>
<p><a href="rescue90.pdf">90-page PDF</a> · <a href="medicine/medicine_30cases.pdf">Medicine PDF</a> · <a href="grozi/grozi_30cases.pdf">GroZi PDF</a> · <a href="isic/isic_30cases.pdf">ISIC PDF</a> · <a href="data/cases.csv">CSV table</a> · <a href="README.md">Readme</a></p>
<nav><button onclick="show('all')">All 90</button><button onclick="show('medicine')">Medicine 30</button><button onclick="show('grozi')">GroZi 30</button><button onclick="show('isic')">ISIC 30</button></nav>'''
    (OUT/'index.html').write_text(header+'\n'.join(cards)+"<script>function show(d){document.querySelectorAll('article').forEach(x=>x.hidden=d!=='all'&&x.dataset.dataset!==d)}</script></html>")
    (OUT/'README.md').write_text('''# new HYP: 90 successful COST1 corrections — English edition

30 medicine queries, 30 GroZi queries and 30 ISIC queries. Each case has a 3200×1800 PNG, vector SVG and one-page PDF. The package also contains a 90-page PDF, three 30-page PDFs, contact sheets, an offline HTML gallery, CSV/JSON decisions, native FP64 heatmaps and image thumbnails. No numbered fold labels appear in the figures. Both references use enlarged full-column panels, approximately twice the original displayed size, below their corresponding heatmaps.

## Selection and model provenance

- Medicine: the original H593 grouped held-out COST1 predictions, RAW 426→481/593 (58 rescues, 3 breaks). Excluding medicine queries already in the original 18 figures leaves 48 eligible rescues. We include all 12 remaining DIFFICULT cases and one NDV2 case, then 17 OUTCOME cases, prioritizing new identities. The 30 queries cover 24 identities. The exact source fold, head parameters and training exclusion remain in the per-case JSON.
- GroZi: frozen full-H593 COST1, RAW 321→355/480 (34 rescues, no breaks), with no GroZi training or calibration. The 30 selected queries cover 23 product identities. Query images are the dataset product crops used by the original experiment. The natural C128 comes from the original mixed gallery, so an incorrect reference can be a legacy medicine image.
- ISIC: frozen full-H593 COST1, RAW 466→500/537 (34 rescues, no breaks). We exclude the previously shown ISIC-Q-0013 and select 30 queries spanning 26 lesions and 25 patients. This is an opened exploratory same-lesion retrieval panel, not a clinical diagnosis experiment or a new untouched confirmation set. Original ISIC IDs, attribution and existing per-image license metadata are preserved for query, incorrect reference and correct reference.

Success is defined by the sealed retrieval identity labels, not by a new visual judgment. The original 18 figures and experimental files are unchanged. This selected collection does not estimate overall accuracy.

## Reading the figures

The same query is shown alongside its cached query-token visibility grids for the RAW incorrect reference and the correct reference. S is the original weighted content score. M = sqrt(mean(query visibility) × mean(reference visibility)). The COST1 head combines several signals; neither S nor M alone is the final action score. The grids retain the original FP64 values and native shape, use nearest-pixel display and a fixed 0–1 scale, and are not segmentation masks or ownership evidence.

Each case retains all 127 sealed challenger logits plus HOLD=0, the physical candidate axis, head provenance and both query/reference visibility arrays. The frozen external head JSON is in data/heads. Source identifiers and license records are preserved verbatim; presentation labels are in English.

## Reproduction

source/render_rc_cost1_rescue90_english_v1.py redraws the verified data without encoder or RoMa inference, retraining, candidate changes or new experiment results. The English figures use the same arrays, decisions and image thumbnails as the verified source collection. The JSON validation receipts document byte comparisons and independent checks.
''')
    v=copy.deepcopy(base_validation)
    v.update(status='COST1_RESCUE90_ENGLISH_RENDER_PASS',language='English',source_validation=bind(BASE/'validation.json'),
             source_independent_validation=bind(BASE/'independent_validation.json'),all_visible_captions_english=True,
             source_arrays_byte_identical=True,source_decisions_unchanged=True,
             enlarged_reference_panels=True,reference_panel_size_fraction=[.235,.235],
             original_reference_panel_size_fraction=[.093,.111])
    dump(OUT/'validation.json',v)
    dump(OUT/'source_manifest.json',dict(base_cases=bind(BASE/'cases_manifest.json'),renderer=bind(Path(__file__)),
         base_validation=bind(BASE/'validation.json'),source_arrays_and_photos='Exact copies from the independently verified source collection'))
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ('.zip','.log','.out','.err','.partial') and p.name!='archive_validation.json']
    archive=OUT/'new_HYP_rescue90_English_complete.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read(str(p.relative_to(OUT)))).hexdigest()==sha(p)
    dump(OUT/'archive_validation.json',dict(status='ZIP_LOOSE_FILES_BYTE_IDENTICAL',files=len(files),archive=str(archive),
         bytes=archive.stat().st_size,sha256=sha(archive)))
    print(json.dumps(dict(status=v['status'],cases=90,archive=str(archive),bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':main()
