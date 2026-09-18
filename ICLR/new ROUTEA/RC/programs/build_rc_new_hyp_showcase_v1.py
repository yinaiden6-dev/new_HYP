#!/usr/bin/env python3
"""Build a Chinese presentation from sealed results; no fitting or inference."""
from __future__ import annotations
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/figures/new_hyp_showcase_20260915_v1'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'data').mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/new_hyp_showcase_mpl_20260915')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps, ImageDraw

FONT = '/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'], 'font.size': 15,
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.edgecolor': '#D5DEE7', 'axes.labelcolor': '#24374B', 'xtick.color': '#546477',
    'ytick.color': '#546477', 'savefig.facecolor': '#FAFCFE'})
INK, MUTED, TEAL, BLUE, GOLD, RED, GRAY = '#142B42', '#586B7D', '#087F8C', '#507DAD', '#C18B32', '#BA4B56', '#A8B5C5'
BG, PALE, RULE = '#FAFCFE', '#EDF4F7', '#DCE5ED'
COLORS = {'RAW': GRAY, 'COST4': BLUE, 'COST1': TEAL, 'CE': GOLD}
SOURCES, SLIDES, CASES, TABLES = {}, [], [], {}
COMMON = '固定 full-H593 商品头 · RAW 自然 C128 · 127 challenger · HOLD/SWITCH · 零外部训练/校准 · GroZi/RPC外部确认，ISIC探索'
SPECS = [
    dict(key='GroZi', folder='rc_new_hyp_grozi120_external_v1', n=480, scope='外部确认', unit='27 个来源视频',
         ci='video_bootstrap95', mean='equal_video_difference', gallery_label='旧图库5413物理条目 + 120新reference',
         sha='0bee519b388b4e54209a843199ba34285282ec95ec5941ce307b9b9165e426fa'),
    dict(key='RPC', folder='rc_new_hyp_rpc_transfer_v1', n=600, scope='外部确认', unit='200 SKU，17类别内分层',
         ci='sku_stratified_bootstrap95', mean='equal_sku_difference', gallery_label='独立200-reference图库',
         sha='6679415e8301b3b80dd49e57b70ebb7d48606659d294869c79e28fedb486bd08'),
    dict(key='ISIC', folder='rc_new_hyp_isic_transfer_v1', n=537, scope='已打开队列探索', unit='346 位患者',
         ci='patient_bootstrap95', mean='equal_patient_difference', gallery_label='独立390-reference图库',
         sha='6af5bfbafd36f9438372b2c5c1fff3e2f154ffa9c9af84000278809b14ea0d46')]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def source(p):
    p = Path(p)
    if not p.is_absolute(): p = ROOT / p
    key = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    if key not in SOURCES: SOURCES[key] = dict(sha256=sha(p), bytes=p.stat().st_size)
    return p

def read(p): return json.loads(source(p).read_text())

def pair(rows, a, b):
    return (sum(not r['correct'][a] and r['correct'][b] for r in rows),
            sum(r['correct'][a] and not r['correct'][b] for r in rows))

def write_csv(name, rows):
    TABLES[name] = rows
    with (OUT / 'data' / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def load_results():
    metrics, comparisons = [], []
    for s in SPECS:
        s['work'] = ROOT / 'results' / s['folder']
        p = s['work'] / 'result.json'
        assert sha(p) == s['sha']
        d = read(p); s['result'] = d
        assert len(d['rows']) == len({r['query_id'] for r in d['rows']}) == s['n']
        for filename in ['result_validation.json', 'independent_final_audit_v1.json']:
            v = read(s['work'] / filename)
            assert 'PASS' in v['status']
        for m, count in d['counts'].items():
            assert count == sum(r['correct'][m] for r in d['rows'])
            rr, bb = pair(d['rows'], 'RAW', m)
            assert count - d['counts']['RAW'] == rr - bb
            if m in d['MRR']:
                assert abs(np.mean([1/r['ranks'][m] for r in d['rows']]) - d['MRR'][m]) < 1e-12
            metrics.append(dict(dataset=s['key'], evidence_scope=s['scope'], model=m, queries=s['n'],
                correct=count, accuracy_percent=count / s['n'] * 100, MRR=d['MRR'][m],
                rescue_vs_RAW=rr, break_vs_RAW=bb, net_vs_RAW=rr-bb,
                candidate_source=d['candidate_source'], model_lineage=d['model_lineage'],
                action='127 challengers; SWITCH iff max logit > 0 else HOLD'))
        for key, v in d['comparisons'].items():
            a,b = key.split('__to__'); rr,bb = pair(d['rows'], a, b)
            assert (rr,bb) == (v['rescue'],v['loss']) and rr-bb == v['net']
            # Independently reproduce the estimand; reuse sealed bootstrap bounds.
            groups = {}
            for row in d['rows']:
                groups.setdefault(row['component'], []).append(int(row['correct'][b])-int(row['correct'][a]))
            mean = float(np.mean([np.mean(values) for values in groups.values()]))
            assert abs(mean-v[s['mean']]) < 1e-12
            comparisons.append(dict(dataset=s['key'], evidence_scope=s['scope'], baseline=a, model=b,
                rescue=rr, breaks=bb, net=rr-bb, query_gain_pp=(rr-bb)/s['n']*100,
                grouped_mean_gain_pp=mean*100, grouped_CI95_low_pp=v[s['ci']][0]*100,
                grouped_CI95_high_pp=v[s['ci']][1]*100, unit=s['unit'],
                reliable_positive=v['reliable_positive'], bootstrap_source='sealed result; no resampling for plotting'))
        s['workers'] = {x['query_id']: x for x in read(s['work']/'worker_manifest.json')['records']}
        gallery = 'gallery_append_manifest.json' if s['key']=='GroZi' else 'gallery_manifest.json'
        s['gallery'] = {x['physical_row']: x for x in read(s['work']/gallery)['records']}
    write_csv('all_models', metrics); write_csv('paired_comparisons', comparisons)
    source(ROOT/'plan/RC_NEW_HYP_PAPER_SCOPE_FREEZE_V1_20260915.md')

def wrapped(s, width=34):
    # Count CJK as two columns for predictable line wrapping in portrait notes.
    lines=[]
    for paragraph in s.split('\n'):
        line=''; n=0
        for c in paragraph:
            k=2 if ord(c)>255 else 1
            if n+k>width and line: lines.append(line); line=''; n=0
            line+=c; n+=k
        lines.append(line)
    return '\n'.join(lines)

def canvas(title, subtitle, source_label='当前冻结结果与实现', footer=COMMON):
    fig = plt.figure(figsize=(16,9), facecolor=BG)
    fig.text(.05,.946,'new HYP  /  RESEARCH BRIEFING',fontsize=10,color=TEAL,weight='bold')
    fig.text(.05,.882,title,fontsize=28,color=INK,weight='bold')
    fig.text(.05,.832,subtitle,fontsize=14,color=MUTED)
    fig.add_artist(plt.Line2D([.05,.95],[.805,.805],transform=fig.transFigure,color=RULE,lw=1))
    fig.text(.05,.066,footer,fontsize=10,color=MUTED)
    fig.text(.05,.030,'2026-09-15  ·  来源：'+source_label,fontsize=9,color=MUTED)
    fig.text(.95,.030,f'{len(SLIDES)+1:02d}',ha='right',fontsize=11,color=TEAL)
    return fig

def text(fig,x,y,s,size=17,color=INK,**kw):
    return fig.text(x,y,s,fontsize=size,color=color,va='top',**kw)

def box(fig,x,y,w,h,title,body='',accent=TEAL,size=18):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.014',
        transform=fig.transFigure,facecolor='white',edgecolor=RULE,lw=1)
    fig.add_artist(p)
    fig.add_artist(Rectangle((x+.014,y+h-.049),.025,.004,transform=fig.transFigure,color=accent))
    text(fig,x+.022,y+h-.075,title,size,accent,weight='bold')
    if body:text(fig,x+.022,y+h-.125,body,14,MUTED,linespacing=1.65)

def arrow(fig,x1,y1,x2,y2,color=TEAL):
    fig.add_artist(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=16,
        transform=fig.transFigure,color=color,lw=1.8))

def finish(fig, stem, note, takeaway):
    fig.canvas.draw()
    # Detect text outside the page before exporting.
    renderer=fig.canvas.get_renderer(); w,h=fig.canvas.get_width_height()
    overflow=[]
    hidden_ticks=set()
    for ax in fig.axes:
        for labels,positions,limits in [(ax.get_xticklabels(),ax.get_xticks(),ax.get_xlim()),
                                        (ax.get_yticklabels(),ax.get_yticks(),ax.get_ylim())]:
            for label,position in zip(labels,positions):
                if not min(limits)-1e-9 <= position <= max(limits)+1e-9:hidden_ticks.add(id(label))
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text() or id(artist) in hidden_ticks:continue
        b=artist.get_window_extent(renderer)
        if b.width and (b.x0 < -2 or b.y0 < -2 or b.x1 > w+2 or b.y1 > h+2):
            overflow.append(artist.get_text()[:80])
    assert not overflow, (stem,overflow)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/(stem+'.'+ext),dpi=200)
    SLIDES.append(dict(stem=stem,title=fig.texts[1].get_text(),note=note,takeaway=takeaway))
    DECK.savefig(fig)
    plt.close(fig)
    print('RENDERED ' + stem, flush=True)

def photo(fig, rect, path, label='',color=INK):
    source(path)
    with Image.open(path) as im: image=ImageOps.exif_transpose(im).convert('RGB'); image.thumbnail((1300,1000))
    ax=fig.add_axes(rect); ax.imshow(image); ax.set_axis_off()
    if label:ax.set_title(label,fontsize=13,color=color,pad=9)
    return ax

def setup_cases():
    for s in SPECS:
        rows=sorted(s['result']['rows'],key=lambda z:z['query_id'])
        good=[z for z in rows if not z['correct']['RAW'] and z['correct']['COST1'] and
              z['selected']['RAW'] in s['gallery'] and z['selected']['COST1'] in s['gallery']]
        assert good
        s['case']=good[0]
        for kind, selected in [('rescue',good[:1]),('break',[z for z in rows if z['correct']['RAW'] and not z['correct']['COST1']][:1]),
            ('remaining_error',[z for z in rows if not z['correct']['COST1'] and z['target_in_C128']][:1])]:
            if kind=='remaining_error' and s['key']!='ISIC':continue
            for z in selected:
                CASES.append(dict(dataset=s['key'],kind=kind,query_id=z['query_id'],
                    selection_rule='lexicographically first in outcome category; rescue requires displayed refs in dataset gallery',
                    RAW_correct=z['correct']['RAW'], COST1_correct=z['correct']['COST1'],
                    RAW_selected=z['selected']['RAW'], COST1_selected=z['selected']['COST1'],target=z['identity'],
                    query_image=str(s['workers'][z['query_id']]['image_path'])))
    write_csv('illustrative_cases', CASES)

def extract_encoding():
    import torch
    torch.set_num_threads(2)
    s=SPECS[1];row=s['case'];worker=s['workers'][row['query_id']]
    shard=worker['execution_ordinal']//8
    def load(stage):
        folder=s['work']/stage/f'shard{shard:02d}'
        rec=read(folder/'receipt.json');path=source(rec['payload']['path'])
        assert sha(path)==rec['payload']['sha256']
        return torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    raw=load('raw');roma=load('roma')
    q=next(x for x in raw['records'] if x['query_id']==row['query_id'])
    m=next(x for x in roma['records'] if x['query_id']==row['query_id'])
    ref=raw['references'][row['selected']['COST1']]
    values={'tokens':ref['tokens'].float().numpy(),'reference_grid':np.array(ref['grid_shape']),
        'query_grid':np.array(q['query_grid_shape'])}
    cases=[]
    for role,key in [('wrong','RAW'),('target','COST1')]:
        c=next(x for x in m['candidates'] if x['physical_row']==row['selected'][key])
        values[role+'_query_visibility']=c['query_visibility'].numpy().reshape(q['query_grid_shape'])
        cases.append(dict(role=role,physical_row=c['physical_row'],scores=c['old_scores']))
    np.savez_compressed(OUT/'data/encoding_example.npz',**values)
    info=dict(query_id=row['query_id'],reference_id=s['gallery'][row['selected']['COST1']]['reference_id'],
        reference_path=ref['source_path'],query_path=worker['image_path'],reference_grid=ref['grid_shape'],
        query_grid=q['query_grid_shape'],token_shape=list(ref['tokens'].shape),
        reference_frame='decoded raw before EXIF; shown encoding-grid figure uses same frame',
        visibility_display='native query token grid, not a segmentation or ownership map',candidate_evidence=cases)
    (OUT/'data/encoding_example.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    return info,values

def build_slides():
    fig=canvas('从参考照片到可靠纠错','new HYP · 主 COST1 · GroZi / RPC 外部确认；ISIC 已打开队列探索',footer='相同 full-H593 固定商品头，相对各自 RAW 自然C128。ownership、完整过目不忘系统与新编码器训练为未来工作。')
    box(fig,.06,.34,.27,.39,'01  用照片登记身份','reference 保存视觉内容\n共享模型比较新 query\n无需为每个新身份训练分类头')
    box(fig,.365,.34,.27,.39,'02  学会何时纠错','内容相容性 × 匹配质量\n检索标签训练七参数小头\nCOST1 改善过度保守的决策')
    box(fig,.67,.34,.27,.39,'03  用独立数据检验','GroZi：321 → 355 / 480\nRPC：192 → 207 / 600\nISIC：466 → 500 / 537')
    text(fig,.075,.22,'“有相似局部”只是证据；是否支持同一身份，需要在候选间联合比较。',21)
    finish(fig,'01_overview','开场先定义任务：我们要认出同一个具体物体或病灶，而不是只判断商品或疾病类别。reference是登记照片。当前方法把图像匹配提供的质量信息，与内容编码器提供的相似性结合，再学习何时修改基础检索结果。右侧三组都是同一固定COST1商品头相对各自RAW的结果。GroZi和RPC属于预定范围的外部确认；ISIC是已打开队列的跨域探索，三者不合并统计。','先说明任务、机制和证据范围，再介绍模型名称。')

    enc,arr=extract_encoding();ng,dim=enc['token_shape'];gh,gw=enc['reference_grid']
    fig=canvas('Reference 如何变成可检索的记忆','真实 RPC reference 的编码示例：局部视觉 token + 位置网格 + 身份索引',source_label='固定编码程序与 RPC 缓存；data/encoding_example.json',footer='冻结 ColNomic-7B；128 是每个 token 的维数，C128 是检索候选数。网格是编码位置，不是物体分割。')
    with Image.open(enc['reference_path']) as im:image=im.convert('RGB');image.thumbnail((1100,1100))
    ax=fig.add_axes([.075,.31,.24,.42]);ax.imshow(image);ax.axis('off')
    for x in np.linspace(0,image.width,gw+1):ax.axvline(x,color='white',alpha=.35,lw=.35)
    for y in np.linspace(0,image.height,gh+1):ax.axhline(y,color='white',alpha=.35,lw=.35)
    ax.set_title(f'登记照片 / {enc["reference_id"]}',fontsize=14)
    arrow(fig,.33,.52,.40,.52)
    box(fig,.415,.33,.22,.37,'冻结编码器',f'RGB 与固定预处理\n上下文视觉表示\n投影并归一化到 128 维\n{gh} × {gw} = {ng} 个视觉 token',size=18)
    arrow(fig,.646,.52,.68,.52)
    ax=fig.add_axes([.72,.34,.22,.36]);lim=float(np.quantile(np.abs(arr['tokens']),.99))
    ax.imshow(arr['tokens'],aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim,rasterized=True)
    ax.set_title(f'实际 token 矩阵：{ng} × {dim}',fontsize=14);ax.set_xlabel('学习到的向量维度',fontsize=12);ax.set_ylabel('位置 token',fontsize=12)
    ax.tick_params(labelsize=10)
    text(fig,.08,.21,'图库保存这些表示及原图路径；RoMa 在 query 到来后读取图像对，生成动态匹配证据。',17)
    finish(fig,'02_reference_encoding',f'这页展示真实RPC缓存中的reference。它的网格为{gh}乘{gw}，共有{ng}个图像token，每个128维。右图就是保存的数值矩阵，颜色只表示向量分量，不能把某一维直接命名为文字或颜色属性。每个token包含编码器上下文，并非独立裁图特征。RAW阶段还保留固定processor模板token；局部加权评分只读取图像token。reference登记不训练身份专属参数，也不输入身份编号、mask或诊断文字。','reference 是一组可复用视觉表示；配对证据在 query 到来后产生。')

    fig=canvas('推理的完整路径','固定图库检索 → 候选条件证据 → 共享比较 → 一次 HOLD / SWITCH',source_label='run_rc_new_hyp_isic_inference_v1.py 与原冻结评分图')
    box(fig,.06,.55,.19,.18,'Query 照片','同一冻结内容编码器',size=17)
    box(fig,.06,.29,.19,.18,'Reference 图库','每个身份的登记照片',size=17)
    arrow(fig,.255,.64,.30,.64);arrow(fig,.255,.38,.30,.38)
    fig.add_artist(plt.Line2D([.278,.278],[.38,.64],transform=fig.transFigure,color=TEAL,lw=1.6))
    box(fig,.31,.54,.24,.20,'全库 RAW MaxSim','每个 query token\n搜索完整 reference token',size=17)
    box(fig,.31,.28,.24,.20,'RoMa 两图配对','只对 C128 内候选配对\n产生候选条件的双侧软可见性',size=17)
    arrow(fig,.43,.53,.43,.49)
    arrow(fig,.56,.64,.61,.64);arrow(fig,.56,.38,.61,.38)
    box(fig,.62,.54,.30,.20,'自然 C128','保留 RAW 首选\n其余 127 个作为 challenger',size=17)
    box(fig,.62,.28,.30,.20,'联合证据 + 七参数头','最高 challenger logit > 0：SWITCH\n否则 HOLD，保持 RAW 首选',size=17)
    arrow(fig,.775,.53,.775,.49)
    text(fig,.07,.19,'Reference 内容是静态资源；可见性与候选比较是每次 query 的动态关系。',19)
    finish(fig,'03_runtime_mechanism','按箭头讲清两条路径。RAW用全库内容相似性提出128个候选；RoMa对query和每个候选reference做配对，提供双侧软权重。内容匹配仍可在完整reference内寻找最相似token，没有变成硬坐标单点绑定。共享小头比较每个challenger与RAW首选，最终最多切换一次。HOLD只表示保留原答案，不是未知对象拒识。这里只使用冻结模型前向，没有根据测试标签选候选。','质量信息参与内容读取，决策同时保留 RAW 先验。')

    fig=canvas('质量与内容怎样联合起来','图像相似性提供内容支持；候选配对提供软权重；共享头读取相对证据',source_label='原 FP64 weighted MaxSim 与 candidate_feature')
    text(fig,.07,.74,r'$S_g=M_gL_g$',32,TEAL)
    text(fig,.07,.63,'M：两侧平均可见性的几何平均\nL：query 权重下的 full-reference\n      weighted MaxSim 平均相似度',18,linespacing=1.6)
    text(fig,.07,.40,'关键：每个 query token\n仍可搜索整张 reference。',22,TEAL,weight='bold')
    labels=[['1','RAW 分差','challenger 原本落后多少'],['2','局部联合得分 S','内容与质量共同提供多少支持'],['3','可见性量 M','有多少配对支持'],['4','归一化内容 L','支持部分的匹配强度'],['5','query 权重移位差','对 query 权重位置的敏感性'],['6','reference 权重移位差','对 reference 权重位置的敏感性']]
    ax=fig.add_axes([.46,.23,.48,.51]);ax.axis('off')
    t=ax.table(cellText=labels,colLabels=['','相对特征','含义'],cellLoc='left',colWidths=[.07,.41,.52],bbox=[0,0,1,1]);t.auto_set_font_size(False);t.set_fontsize(12.5)
    style_table(t)
    text(fig,.075,.18,'输出 z = 六个特征的加权和 + 偏置；所有候选共享 7 个参数。',18)
    finish(fig,'04_joint_evidence','M与L是当前评分图的因子化描述：S等于M乘L。M反映两侧软可见性总量，L是带reference权重的内容匹配平均值。头输入是challenger相对RAW首选的六项特征，含RAW分差以及质量、内容和移位控制。虽然最终头是线性的，输入已经包含乘积、最大匹配和相对归一化等非线性计算。移位特征的存在本身不证明ownership。','小头比较的是质量与内容的联合证据，不是只看一个相似性数值。')

    fig=canvas('训练改变了什么：COST4 → COST1','编码器、匹配方式、六项特征、头结构和推理阈值保持一致',source_label='run_rc_six_cause_loss_binding_v1.py；固定 full-H593 head 清单',footer='任务新增监督只有 query-reference 身份关系；基础模型预训练来源单独披露。外部数据不用于训练或阈值选择。')
    box(fig,.065,.36,.27,.37,'训练输入','query 与正确 reference\n自然检索产生的错误候选\n不使用任务内框、mask 或对应点')
    box(fig,.365,.36,.27,.37,'唯一损失改动','压制错误 challenger 的系数\n4  →  1\n重新学习共享的七个参数')
    box(fig,.665,.36,.27,.37,'保持相同推理规则','最高 challenger 分数 > 0\n才切换，否则保持 RAW\n不是在外部测试集调阈值')
    text(fig,.075,.24,'“正确候选已经排在 challenger 第一”与“模型愿意切换过去”是两个条件。',20)
    finish(fig,'05_retrieval_only_training','retrieval-only指我们新增的任务监督只需要query-reference身份关系，不需要逐图空间标注。训练头学会正确候选超过HOLD和竞争候选，同时控制错误切换。COST4对错误challenger施加4倍压制，COST1改为1倍，其余训练结构沿用冻结协议。这个改动在更广开发数据和跨数据集上有收益。它没有训练新的token编码器，也不能概括为所有历史错误都只是过度保守。','优化的是共享决策规则；reference 编码方式沿用原实现。')

    fig=canvas('同一固定商品头，在三个数据集上评测','主模型 COST1；CE 为预定次模型。每个面板分别与自己的 RAW / COST4 比较。')
    for i,s in enumerate(SPECS):
        ax=fig.add_axes([.075+i*.305,.25,.255,.47]);models=['RAW','COST4','COST1','CE'];d=s['result'];counts=[d['counts'][m] for m in models]
        ax.bar(range(4),np.array(counts)/s['n']*100,color=[COLORS[m] for m in models],width=.65)
        ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('正确率（%）' if i==0 else '',fontsize=13)
        ax.set_xticks(range(4),models,fontsize=12);ax.set_title(f'{s["key"]} · {s["scope"]}',fontsize=16,pad=15)
        ax.grid(axis='y',color=RULE);ax.set_axisbelow(True)
        for j,c in enumerate(counts):ax.text(j,c/s['n']*100+2,f'{c}/{s["n"]}\n{c/s["n"]*100:.1f}%',ha='center',fontsize=11,color=INK)
        ax.text(.5,-.21,s['gallery_label'],transform=ax.transAxes,ha='center',fontsize=10,color=MUTED)
    finish(fig,'06_accuracy','这张图的准确率纵轴统一从零到百分之百，并在每根柱上标出分子分母。COST1从各自RAW的321、192、466，提升到355、207、500。CE结果也完整展示，但仍是预定次模型。不同数据集的图库、任务和抽样不同，不把它们连成样本扩大曲线，不以准确率高低推断难度，也不合并为一个总分。','跨数据集收益成立；各面板保留各自的任务和分母。')

    fig=canvas('主模型 COST1：救回与损失同时报告','正方向表示救回；负方向表示原正确变错。零损失是观察结果，不是重设资格门。')
    ax=fig.add_axes([.19,.23,.68,.50]);ys=[];labs=[]
    for i,s in enumerate(SPECS):
        for j,base in enumerate(['RAW','COST4']):
            y=5-(i*2+j);rr,bb=pair(s['result']['rows'],base,'COST1')
            ax.barh(y,rr,color=TEAL,height=.57);ax.barh(y,-bb,color=RED,height=.57)
            ax.text(rr+.6,y,f'{rr} 救 / {bb} 损，净增 +{rr-bb}',va='center',fontsize=13,color=INK)
            ys.append(y);labs.append(f'{s["key"]}{s["n"]} / {base}')
    ax.axvline(0,color=INK,lw=1);ax.set_xlim(-7,51);ax.set_yticks(ys,labs,fontsize=13);ax.set_xlabel('query 数',fontsize=13);ax.grid(axis='x',color=RULE);ax.set_axisbelow(True)
    finish(fig,'07_rescue_break','救回和损失必须成对报告。相对RAW，GroZi是34救0损，RPC是18救3损，ISIC是34救0损。相对相同数据同结构的COST4，分别是29救0损、16救3损、31救0损。RPC说明当前主张是可靠群体净增，不保证每一张原正确都保持。新模型的SWITCH还可能是错误换成另一错误，所以切换次数不能直接算救回次数。','准确率增益需要展开为救回与损失，不能只报告切换次数。')

    fig=canvas('Reference 绑定：证据必须属于当前候选','保持 RAW 与模型参数，打乱候选与整组配对证据的关联，检验身份绑定的作用。')
    for i,s in enumerate(SPECS):
        ax=fig.add_axes([.085+i*.305,.25,.245,.46]);d=s['result'];c=[d['counts']['COST1_CBIND'],d['counts']['COST1']]
        ax.bar([0,1],np.array(c)/s['n']*100,color=[GRAY,TEAL],width=.58)
        ax.axhline(d['counts']['RAW']/s['n']*100,color=BLUE,ls='--',lw=1.2,label='RAW')
        ax.set_ylim(0,108);ax.set_xticks([0,1],['绑定打乱','正确绑定'],fontsize=13);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',color=RULE);ax.set_axisbelow(True)
        ax.set_title(s['key']+' · COST1 · '+('探索' if s['key']=='ISIC' else '外部'),fontsize=16,pad=16)
        for j,v in enumerate(c):ax.text(j,v/s['n']*100+3,f'{v}/{s["n"]}',ha='center',fontsize=15,color=INK)
        if i==0:ax.legend(frameon=False,fontsize=11,loc='upper left')
    text(fig,.08,.17,'这是候选身份绑定证据；不等同于像素归属、空间定位或 ownership 的证明。',17)
    finish(fig,'08_candidate_binding','CBIND把候选关联的整组配对证据移给另一候选，保持RAW候选轴和头参数。COST1打乱后分别为313、174、414，完整绑定为355、207、500。这个干预说明正确reference与证据的关联在当前系统中有作用。它并不隔离每一个token位置，也不意味着我们已经找到了目标的真实mask。','当前建立的是候选身份层面的证据绑定。')

    fig=canvas('群体可靠性：按相关样本的组来统计','固定 full-H593 COST1 相对 RAW / COST4 的分组平均增益及封存的 95% bootstrap 区间。',footer='GroZi/RPC外部确认，ISIC探索。区间按视频/类别内SKU/患者计算，未重采样。各自自然RAW C128；不是微平均柱的误差条。')
    ax=fig.add_axes([.28,.23,.57,.50]);ys=[];labs=[];cirows=[]
    for i,s in enumerate(SPECS):
        for j,base in enumerate(['RAW','COST4']):
            v=s['result']['comparisons'][base+'__to__COST1'];mean=v[s['mean']]*100;lo,hi=np.array(v[s['ci']])*100;y=5-i*2-j
            ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color=TEAL if j==0 else BLUE,capsize=5,ms=8,lw=2)
            ax.text(hi+.18,y,f'{mean:.2f} [{lo:.2f}, {hi:.2f}]',va='center',fontsize=11,color=INK)
            ys.append(y);labs.append(f'{s["key"]} / {base}\n{s["unit"]}')
    ax.axvline(0,color=RED,ls='--');ax.set_xlim(-.25,14.5);ax.set_yticks(ys,labs,fontsize=11);ax.set_xlabel('分组平均准确率增益（百分点）',fontsize=13);ax.grid(axis='x',color=RULE);ax.set_axisbelow(True)
    finish(fig,'09_grouped_uncertainty','不能把同一视频、同一商品或同一患者的照片全部视为独立样本。这里复用原方案封存的分组bootstrap区间。显示的是等组平均的配对准确率差，和前面的query微平均不是同一个估计量。主COST1相对RAW和COST4的这些区间下界均大于零。CE相对COST1在GroZi和RPC的区间跨零，因此它没有取代预定主模型。','可靠净增需要考虑组内相关性，而不仅是多出几张正确。')

    s=SPECS[2];rows=[r for r in s['result']['rows'] if not r['correct']['COST4'] and r['correct']['COST1']]
    assert len(rows)==31 and all(r['target_top_challenger_held']['COST4'] for r in rows)
    preds={}
    for p in sorted((s['work']/'predictions').glob('shard*/payload.json')):
        v=read(p)
        for z in v['records']:preds[z['query_id']]=z
    logits=[]
    for r in rows:
        p=preds[r['query_id']];z4=np.array([float.fromhex(v) for v in p['models']['COST4']['logits_hex']]);z1=np.array([float.fromhex(v) for v in p['models']['COST1']['logits_hex']])
        assert z4.max()<=0 and z1.max()>0 and z4.argmax()==z1.argmax()
        logits.append(dict(query_id=r['query_id'],COST4_max_logit=float(z4.max()),COST1_max_logit=float(z1.max()),same_top_challenger=True))
    write_csv('ISIC31_held_to_rescue',logits)
    fig=canvas('一次明确的失效修复：找到 target，却没有切换','ISIC 中 COST1 相对 COST4 新救回的全部 31 张；旧头的最高 challenger 已经是 target。',source_label='ISIC result rows + sealed prediction logits；逐行重新核对',footer='仅展示这31张事后定义的救回子集；不是整个人群分布。两个头都用原固定阈值0，无ISIC训练或调参。')
    ax=fig.add_axes([.07,.23,.52,.49]);rr=sorted(logits,key=lambda z:z['COST4_max_logit']);x=np.arange(1,32)
    for j,z in enumerate(rr):ax.plot([x[j],x[j]],[z['COST4_max_logit'],z['COST1_max_logit']],color=RULE,lw=2)
    ax.scatter(x,[z['COST4_max_logit'] for z in rr],color=BLUE,label='COST4：HOLD',s=34)
    ax.scatter(x,[z['COST1_max_logit'] for z in rr],color=TEAL,label='COST1：SWITCH 正确',s=34)
    ax.axhline(0,color=RED,ls='--',lw=1.5);ax.set_xlabel('31 个救回 query（按旧分数排序）',fontsize=12);ax.set_ylabel('最高 challenger logit',fontsize=12);ax.legend(frameon=False,fontsize=11)
    box(fig,.65,.46,.27,.26,'31 / 31','旧头已找到正确 challenger\n旧分数 ≤ 0，新分数 > 0\n同一原图、候选和视觉证据',size=30)
    text(fig,.67,.37,'COST1 总计：34 救 / 0 损（相对 RAW）\n这31张是相对 COST4 的新增纠错。',15,linespacing=1.6)
    finish(fig,'10_held_target_rescues','这是最具体的机制解释。我们从原始封存logit逐行核对了31个query。对这些照片，COST4和COST1的最高challenger相同，并且就是target；区别在于旧头最高输出不大于零，所以HOLD，新头大于零，所以切换正确。不是重新给模型补入正确reference，也不是改变阈值。这只解释这31次改进，不能推广成所有错误的唯一原因。','部分瓶颈位于从“候选最好”到“愿意纠错”的决策环节。')

    fig=canvas('真实图片：成功纠错长什么样','每个数据集按固定规则选取一个实例；展示 query、RAW 选中 reference 和 COST1 选中 reference。',source_label='封存结果 selected 字段 + 原始数据图像',footer='事后说明性案例，不是随机代表性样本或新的统计检验。GroZi使用数据集商品裁图；RPC整图；ISIC皮肤镜图。')
    for i,s in enumerate(SPECS):
        z=s['case'];y=.56-i*.19
        text(fig,.055,y+.13,s['key'],16,TEAL,weight='bold');text(fig,.055,y+.085,z['query_id'],10,MUTED)
        for j,(label,path,color) in enumerate([
            ('Query',s['workers'][z['query_id']]['image_path'],INK),
            ('RAW：错误 reference',s['gallery'][z['selected']['RAW']]['image_path'],RED),
            ('COST1：正确 reference',s['gallery'][z['selected']['COST1']]['image_path'],TEAL)]):
            photo(fig,[.21+j*.25,y,.21,.15],path,label,color)
    finish(fig,'11_real_rescue_cases','这页只描述模型输出，不凭外观编造错误原因。GroZi案例从已登记120个reference内可展示的救回案例按query编号选第一例；RPC与ISIC按同一规则选第一例。左侧是query，中间是RAW错误选择，右侧是COST1正确选择。请注意reference和query不是同一张文件。对ISIC也只讲同病灶实例检索，不讲疾病诊断。','真实图片把“救回”落到具体 query-reference 决策。')

    fig=canvas('同一 query，面对不同 reference 的质量权重','RPC 成功纠错案例：展示实际缓存的 query 可见性网格，所有热图固定使用 0–1 色标。',source_label='RPC RoMa shard 缓存；data/encoding_example.*',footer='这些是 token 网格上的匹配可见性权重，未叠加成物体mask，也不作为ownership证据。')
    photo(fig,[.07,.33,.25,.38],enc['query_path'],'同一 Query：'+enc['query_id'])
    for j,role in enumerate(['wrong','target']):
        ax=fig.add_axes([.39+j*.29,.33,.23,.38]);im=ax.imshow(arr[role+'_query_visibility'],vmin=0,vmax=1,cmap='viridis',aspect='equal')
        ax.set_title('与 RAW 错误 reference 配对' if role=='wrong' else '与正确 reference 配对',fontsize=13)
        ax.set_xlabel('query token 列',fontsize=11);ax.set_ylabel('query token 行',fontsize=11);ax.tick_params(labelsize=9)
        c=enc['candidate_evidence'][j]['scores'];text(fig,.39+j*.29,.25,f'S = {c["real_score"]:.4f}    M = {c["visibility_mass"]:.4f}',12)
    cb=fig.add_axes([.43,.17,.43,.025]);fig.colorbar(im,cax=cb,orientation='horizontal',ticks=[0,.5,1]);cb.tick_params(labelsize=10)
    finish(fig,'12_actual_visibility','这里展示实际RoMa缓存，不是手工画出的注意力。左边的同一query和两张reference配对后，产生右边两幅query可见性网格。色标固定为零到一，因而颜色可以直接比较。权重高表示当前配对模型认为更有重叠支持，并不保证身份正确；错误reference也可以形成高权重。所以还需要ColNomic内容与共享决策。网格不是分割mask，也没有以目标标签参与计算。','动态关系证据依赖 reference；高可见性本身不等于正确身份。')

    fig=canvas('完整展示边界：仍有损失，也仍有无法救回的 query','各选编号最小的一个 RPC 原正确损失案例与 ISIC 候选内剩余错误案例。',source_label='封存结果 + 原图；选择规则见 illustrative_cases.csv',footer='只展示真实输出与正确reference，不将照片外观猜测写成已证实原因。测试结果未用于训练新头或调整阈值。')
    for i,(s,kind) in enumerate([(SPECS[1],'break'),(SPECS[2],'remaining_error')]):
        c=next(x for x in CASES if x['dataset']==s['key'] and x['kind']==kind)
        z=next(x for x in s['result']['rows'] if x['query_id']==c['query_id']);gt=next(x for x in s['gallery'].values() if x['identity']==z['identity']);y=.46-i*.29
        text(fig,.06,y+.26,f'{s["key"]} / {z["query_id"]}  ·  '+('原正确变错' if kind=='break' else 'target在C128内，仍未认对'),15,RED)
        paths=[s['workers'][z['query_id']]['image_path'],s['gallery'][z['selected']['RAW']]['image_path'],s['gallery'][z['selected']['COST1']]['image_path'],gt['image_path']]
        labels=['Query','RAW选择','COST1选择','正确 reference']
        for j,(path,label) in enumerate(zip(paths,labels)):photo(fig,[.065+j*.235,y,.205,.205],path,label,TEAL if j==3 else INK)
    finish(fig,'13_remaining_failures','这页保留负例，说明我们主张群体净增而不是每张都正确。RPC有3张RAW原正确被COST1破坏，展示其中编号最小的一张。ISIC展示target已经在C128内但COST1仍未认对的一张。因此未解决问题既包含候选召回，也包含候选内的表示、比较和决策问题。我们不根据这一页主观猜测为每张图确定根因。','成功机制和群体增益并不意味着所有剩余问题已经解决。')

    h=read('results/rc_six_cause_isolation_v1/loss_binding/result.json');assert len(h['rows'])==593
    for k,v in h['counts'].items():assert v==sum(z['correct'][k] for z in h['rows'])
    old=read('reports/figures/briefing_new_hyp_20260912_v1/chart_data_audit.json')
    for p,b in old['source_files'].items():assert sha(source(p))==b['sha256']
    history=[]
    for v in old['performance_records'][:3]:
        history.append(dict(panel=v['scope'],head=v['model'],RAW=v['raw'],result=v['correct'],n=v['n'],evidence='historical fixed panel'))
    for k in ['ALL_COST4','GROUP_BASE','COST1','ALL_CE']:
        history.append(dict(panel='H593 grouped OOF5',head=k,RAW=h['counts']['RAW'],result=h['counts'][k],n=593,evidence='development OOF; fold-specific heads'))
    write_csv('historical_lineages',history)
    fig=canvas('历史成绩各自保留；突破不靠更换分母','旧固定面板、H593 开发交叉验证、外部固定头评测属于不同模型与数据协议。',source_label='历史原结果重哈希 + H593 loss_binding result',footer='所有面板均为各自 RAW 自然C128。OOF每折独立训练，外部结果用full-H593固定头；它们不是同一模型的扩样曲线。')
    display=[]
    for v in history:display.append([v['panel'].split(' / ')[0],v['head'],f'{v["RAW"]}/{v["n"]}',f'{v["result"]}/{v["n"]}',f'+{v["result"]-v["RAW"]}'])
    ax=fig.add_axes([.065,.21,.87,.53]);ax.axis('off');t=ax.table(cellText=display,colLabels=['数据面板','对应模型','RAW','模型正确数','净增'],colWidths=[.26,.24,.17,.19,.14],bbox=[0,0,1,1],cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(14);style_table(t)
    finish(fig,'14_history_and_lineage','这一页专门防止历史结果混淆。旧32面板25到28，旧128面板88到99，旧90面板61到69都保留自己的头与协议。H593五折开发结果中COST1是481，CE是486，但这是各折独立训练后的留出预测。外部数据使用重新以全部H593开发数据训练并封存的头，因此不能把外部增益写成旧99/128被突破，也不能把593图上的成绩当作旧128图上的成绩。','所有贡献有明确模型、数据和证据归属。')

    fig=canvas('当前论文收口：机制与增益；未来继续扩展','把已完成的经验支持、工程实现和未来研究目标讲清楚。',source_label='2026-09-15 用户确定的论文范围',footer='new HYP 是当前联合证据校准框架名称。形式化性质与经验验证分别陈述，不宣称普遍正确、零错误或完整过目不忘保证。')
    box(fig,.065,.30,.41,.44,'当前贡献','reference 条件的质量 × 内容联合证据\n任务内仅用检索身份关系训练\n成本校准改进与候选绑定检验\nGroZi / RPC 外部确认 + ISIC 跨域探索',size=23)
    box(fig,.525,.30,.41,.44,'未来方向','Ownership：空间归属与多物体解释\n完整过目不忘：登记、保持、未知拒识\n新编码器：更适合细粒度身份的 token\n冻结 ColNomic 的适配与表达限制',accent=GOLD,size=23)
    text(fig,.08,.20,'贡献落在可检验的机制和技术增益；长期系统目标保留为后续工作。',20)
    finish(fig,'15_scope_and_future','结尾回到用户已经确定的论文边界。当前我们建立并验证的是reference条件下质量和内容的联合校准，以及训练改进带来的群体收益。Ownership作为独立扩展。完整过目不忘系统还需要更完整的登记、图库增长后保持、未知拒识和复杂场景验证。ColNomic是本轮已验证实现，但不是预先认定的最终编码器；新的检索监督编码器作为未来工作，不能提前承诺一定提升。','当前主线可以收口，未来系统目标不再阻塞当前论文。')

def style_table(t):
    for (row,col),cell in t.get_celld().items():
        cell.set_edgecolor('white');cell.set_linewidth(1.8)
        if row==0:cell.set_facecolor(INK);cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        else:cell.set_facecolor(PALE if row%2 else 'white');cell.get_text().set_color(INK)

def create_xlsx():
    from xml.sax.saxutils import escape
    def col(n):
        out=''
        while n:n,k=divmod(n-1,26);out=chr(65+k)+out
        return out
    sheets=list(TABLES)
    with zipfile.ZipFile(OUT/'new_HYP_statistics.xlsx','w',zipfile.ZIP_DEFLATED) as z:
        types='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'+''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1))+'</Types>'
        z.writestr('[Content_Types].xml',types)
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name="{name[:31]}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(sheets,1))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))+'</Relationships>')
        for i,name in enumerate(sheets,1):
            rows=TABLES[name];keys=list(rows[0]);data=[keys]+[[r[k] for k in keys] for r in rows];xml=[]
            for j,row in enumerate(data,1):
                cells=[]
                for k,v in enumerate(row,1):
                    addr=f'{col(k)}{j}'
                    if isinstance(v,(int,float)) and not isinstance(v,bool):cells.append(f'<c r="{addr}"><v>{v}</v></c>')
                    else:cells.append(f'<c r="{addr}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>')
                xml.append(f'<row r="{j}">'+''.join(cells)+'</row>')
            z.writestr(f'xl/worksheets/sheet{i}.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="20" width="23" customWidth="1"/></cols><sheetData>'+''.join(xml)+f'</sheetData><autoFilter ref="A1:{col(len(keys))}{len(data)}"/></worksheet>')

def full_bleed_pptx():
    """Keep the generated speaker notes, give each PNG a full 16:9 slide."""
    import xml.etree.ElementTree as ET
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for key,value in ns.items():ET.register_namespace(key,value)
    target=OUT/'new_HYP_presentation_15slides.pptx'
    with zipfile.ZipFile(target) as z: files={name:z.read(name) for name in z.namelist()}
    root=ET.fromstring(files['ppt/presentation.xml'])
    size=root.find('p:sldSz',ns);size.set('cx','12192000');size.set('cy','6858000');size.set('type','screen16x9')
    files['ppt/presentation.xml']=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    for name in files:
        if not re.fullmatch(r'ppt/slides/slide\d+.xml',name):continue
        root=ET.fromstring(files[name]);tree=root.find('p:cSld/p:spTree',ns)
        pictures=tree.findall('p:pic',ns);assert len(pictures)==1
        for node in list(tree):
            if node.tag in [f'{{{ns["p"]}}}sp',f'{{{ns["p"]}}}graphicFrame']:tree.remove(node)
        pic=pictures[0];sp=pic.find('p:spPr',ns)
        transform=sp.find('a:xfrm',ns)
        if transform is None:transform=ET.SubElement(sp,f'{{{ns["a"]}}}xfrm')
        for node in list(transform):transform.remove(node)
        ET.SubElement(transform,f'{{{ns["a"]}}}off',x='0',y='0')
        ET.SubElement(transform,f'{{{ns["a"]}}}ext',cx='12192000',cy='6858000')
        fill=pic.find('p:blipFill',ns)
        for node in list(fill):
            if node.tag in [f'{{{ns["a"]}}}srcRect',f'{{{ns["a"]}}}stretch',f'{{{ns["a"]}}}tile']:fill.remove(node)
        stretch=ET.SubElement(fill,f'{{{ns["a"]}}}stretch');ET.SubElement(stretch,f'{{{ns["a"]}}}fillRect')
        files[name]=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)

def write_notes():
    intro='''# new HYP 展示讲义

2026-09-15 · 建议讲解时长 15–20 分钟。用途：组会、合作者交流与论文机制说明。

## 一句话介绍

我们用登记reference提供身份依据，把候选条件的匹配质量与内容相容性联合起来，用检索身份监督学习共享纠错规则；同一固定商品训练头在GroZi、RPC取得外部净增，在ISIC取得跨域探索信号。

## 术语速查

- **Reference**：某个具体身份的登记照片。**Gallery**：reference图库。
- **Query**：待识别照片。**Target**：正确身份，只用于训练监督和评测核对。
- **RAW**：基础ColNomic全库MaxSim检索。**C128**：RAW自然检索前128个候选。
- **Challenger**：除RAW首选外的127个候选。**HOLD**：保留RAW答案。**SWITCH**：改选一个challenger。
- **COST4 / COST1**：同一七参数结构，不同错误压制代价。**CE**：预定次模型，候选竞争交叉熵。
- **CBIND**：打乱候选与配对证据的关联，检验正确绑定是否有贡献。
- **Retrieval-only**：本任务新增训练只用身份/检索关系；不表示基础模型全部预训练都无其他监督。

## 完整统计表：相同full-H593固定头

| 数据 | RAW | COST4 | GROUP_COST4 | COST1主 | CE次 | COST1救/损（vs RAW） |
|---|---:|---:|---:|---:|---:|---:|
| GroZi480，外部确认 | 321/480 | 326/480 | 329/480 | 355/480 | 367/480 | 34/0 |
| RPC600，外部确认 | 192/600 | 194/600 | 196/600 | 207/600 | 217/600 | 18/3 |
| ISIC537，已打开队列探索 | 466/537 | 469/537 | 470/537 | 500/537 | 513/537 | 34/0 |

共同推理为自然RAW C128、127 challenger、原阈值0的HOLD/SWITCH。外部零训练或校准。GroZi旧5413物理reference追加120张；RPC独立200-reference；ISIC独立390-reference。所有11个原结果臂及其MRR见Excel与CSV，不合并分母。

## 逐页讲解
'''
    sections=[]
    for i,s in enumerate(SLIDES,1):
        sections.append(f'### {i:02d} {s["title"]}\n\n![{s["title"]}]({s["stem"]}.png)\n\n**本页要点：{s["takeaway"]}**\n\n{s["note"]}\n')
    faq='''
## 常见问题与建议回答

**是不是又训练了大模型？** 这轮没有。ColNomic与RoMa冻结，任务训练更新共享七参数头。新token编码器列为未来研究。

**新身份要重训吗？** 当前结构允许登记reference后沿用共享头；跨数据集实验实际复用了固定商品头。它不保证任何新身份一定被召回或识别，也不等于完整过目不忘。

**理论贡献只是改一个系数吗？** 系数改变是具体技术实现。研究内容是把reference条件的质量、内容和决策关系形式化，隔离评分接口与决策失效，并通过冻结协议、对照和跨数据集检验验证可迁移收益。不能把已有MaxSim或简单代数恒等式称为新发明。

**COST1是不是在测试集选了阈值？** 不是。系数在训练目标里改变，推理仍采用事先固定的logit阈值0。GroZi、RPC、ISIC都不训练或校准头。

**为什么CE数字更高却仍是次模型？** 主次角色在外部结果前已指定。CE相对COST1在GroZi与RPC的分组区间跨零；ISIC次模型对照为正，也不事后改写主模型。

**是否每个旧正确都保住？** 没有这样的普遍保证。RPC COST1有18救3损，主张是可靠群体净增。GroZi和ISIC的零损失是本次观察。

**是不是证明了目标区域/ownership？** 没有。当前是候选身份绑定与联合证据校准；质量网格只是RoMa输出，ownership留作独立扩展。

**ISIC是不是疾病诊断？** 不是。任务是同一病灶的实例检索，且使用已打开并经历史筛选的队列，结论是跨域探索。不同文件/像素哈希不保证独立拍摄会话，也没有排除近重复或未知预训练暴露。

**旧28/32、99/128被突破了吗？** 这批外部数字不能这样解释。旧固定面板成绩保持原有模型与协议；H593是开发OOF，外部为full-H593冻结头。

**ColNomic是唯一瓶颈吗？** 尚未证明。冻结表示的适配性和无法由本轮小头学习新视觉特征是系统限制；候选召回、reference可辨识信息和决策也各有边界。新编码器的收益仍需未来受控实验。

## 展示时的四个提醒

1. 每次报数字同时说出数据集、模型与分母；不把H593 OOF和外部固定头混在一起。
2. 先说主COST1，再完整披露次CE。净增同时报告救回和损失。
3. 图像案例按明确事后规则选取，用于解释输出；不代表整体错误类型的占比。
4. 当前可收口的是限定范围的机制与收益。ownership、完整过目不忘和新编码器属于未来工作。

## 可复现材料

原始结果、程序、图片与缓存的哈希记录在 `source_manifest.json`。全部表格来自原结果逐行复算，bootstrap区间原样读取；本包没有训练、推理、重设阈值或重新抽样。`data/illustrative_cases.csv`说明选图规则，`data/ISIC31_held_to_rescue.csv`列出31次决策变化，`data/encoding_example.*`保存实际编码与质量网格示例。
'''
    (OUT/'new_HYP_lecture_zh.md').write_text(intro+'\n'.join(sections)+faq)
    # Presenter notes remain separate from the slide image in PowerPoint.
    deck='\n\n'.join(f'# {s["title"]}\n\n![]({s["stem"]}.png){{width=100%}}\n\n::: notes\n{s["note"]}\n:::' for s in SLIDES)
    (OUT/'slides_source.md').write_text(deck)
    subprocess.run(['pandoc','slides_source.md','--from=markdown','--to=pptx','--slide-level=1','-o','new_HYP_presentation_15slides.pptx'],cwd=OUT,check=True)
    full_bleed_pptx()
    subprocess.run(['pandoc','new_HYP_lecture_zh.md','--from=markdown','--to=docx','-o','new_HYP_lecture_zh.docx'],cwd=OUT,check=True)
    subprocess.run(['pandoc','new_HYP_lecture_zh.md','--standalone','--metadata','title=new HYP 展示讲义','--to=html5','-o','new_HYP_lecture_zh.html'],cwd=OUT,check=True)
    page=OUT/'new_HYP_lecture_zh.html'
    css='<style>body{max-width:1000px;margin:40px auto;padding:0 28px;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#142b42;line-height:1.85}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #dce5ed}th{background:#edf4f7}h1,h2,h3{color:#087f8c}a{color:#087f8c}@media print{h3{break-before:page}}</style>'
    page.write_text(page.read_text().replace('</head>',css+'</head>'))
    # PDF handout: image + an actual speaking script, not only slide thumbnails.
    with PdfPages(OUT/'new_HYP_lecture_zh.pdf') as pdf:
        for i,s in enumerate(SLIDES,1):
            f=plt.figure(figsize=(8.27,11.69),facecolor='white')
            f.text(.07,.951,f'{i:02d}  {s["title"]}',fontsize=17,color=INK,weight='bold',va='top')
            ax=f.add_axes([.055,.57,.89,.315]);ax.imshow(Image.open(OUT/(s['stem']+'.png')));ax.axis('off')
            f.text(.075,.525,'本页要点',fontsize=14,color=TEAL,weight='bold')
            f.text(.075,.492,wrapped(s['takeaway'],62),fontsize=12,color=INK,va='top',linespacing=1.6)
            f.text(.075,.408,'讲解词',fontsize=14,color=TEAL,weight='bold')
            f.text(.075,.375,wrapped(s['note'],66),fontsize=11.5,color=INK,va='top',linespacing=1.75)
            f.text(.075,.046,'new HYP · 中文讲义 · 2026-09-15     完整FAQ与统计表见同目录 Word / HTML 讲义',fontsize=8,color=MUTED)
            pdf.savefig(f);plt.close(f)

def write_web():
    data=json.dumps(SLIDES,ensure_ascii=False).replace('</',r'<\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>new HYP · 研究展示</title>
<style>*{box-sizing:border-box}body{margin:0;background:#11263a;color:#edf4f7;font-family:system-ui,"Microsoft YaHei",sans-serif}header{padding:18px 4vw;display:flex;gap:12px;align-items:center;flex-wrap:wrap}header strong{margin-right:auto}button,a{font:inherit;color:inherit}button{background:#244158;border:1px solid #567085;border-radius:7px;padding:8px 13px;cursor:pointer}a{color:#89d2d3;text-decoration:none}.layout{max-width:1480px;margin:auto;padding:0 3vw 28px}.stage{background:#fafcfe;border-radius:9px;overflow:hidden;box-shadow:0 12px 40px #0003}.stage img{display:block;width:100%;height:auto}.notes{background:#1c344a;border-radius:9px;padding:18px 24px;margin-top:18px;line-height:1.8}.notes h2{font-size:17px;color:#91d5d7;margin-top:0}.nav{display:flex;gap:10px;justify-content:center;padding:14px}select{max-width:55vw;padding:8px;background:#244158;color:white;border-radius:7px}small{color:#bdcad5}footer{margin:15px 0;font-size:13px;line-height:1.8}@media print{header,.nav,footer{display:none}body{background:white;color:black}.notes{background:white}.stage{box-shadow:none}}</style>
<header><strong>new HYP · 机制、结果与边界</strong><a href="new_HYP_presentation_15slides.pdf">演示 PDF</a><a href="new_HYP_presentation_15slides.pptx">PowerPoint</a><a href="new_HYP_lecture_zh.html">完整讲义</a><a href="new_HYP_statistics.xlsx">统计 Excel</a><button id="toggle">隐藏讲解词</button><button id="full">全屏</button></header>
<main class="layout"><div class="stage"><img id="slide" alt="研究演示页"></div><nav class="nav"><button id="prev">← 上一页</button><select id="select" aria-label="选择演示页"></select><button id="next">下一页 →</button></nav><section class="notes" id="notes"><h2 id="takeaway"></h2><p id="script"></p></section><footer>方向键 / 空格翻页；N 显示讲解词；F 全屏。所有文件可离线打开，无网络请求。<br>主 COST1 / 次 CE；GroZi、RPC 为外部确认，ISIC 为已打开队列探索；不同数据集与历史头不合并分母。</footer></main>
<script>const slides=DATA;let index=0;const $=id=>document.getElementById(id);slides.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=String(i+1).padStart(2,'0')+' · '+s.title;$('select').appendChild(o)});function show(i){index=Math.max(0,Math.min(slides.length-1,i));const s=slides[index];$('slide').src=s.stem+'.png';$('slide').alt=s.title;$('takeaway').textContent=s.takeaway;$('script').textContent=s.note;$('select').value=index;$('prev').disabled=index===0;$('next').disabled=index===slides.length-1}function toggle(){$('notes').hidden=!$('notes').hidden;$('toggle').textContent=$('notes').hidden?'显示讲解词':'隐藏讲解词'}function full(){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen()}$('prev').onclick=()=>show(index-1);$('next').onclick=()=>show(index+1);$('select').onchange=e=>show(Number(e.target.value));$('toggle').onclick=toggle;$('full').onclick=full;document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(['ArrowRight',' ','PageDown'].includes(e.key)){e.preventDefault();show(index+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}if(e.key.toLowerCase()==='n')toggle();if(e.key.toLowerCase()==='f')full()});show(0);</script></html>'''.replace('DATA',data)
    (OUT/'index.html').write_text(page)
    # All-slide contact sheet for quick visual inspection and orientation.
    thumbw,thumbh=640,360;sheet=Image.new('RGB',(thumbw*3,thumbh*5),(235,241,246))
    for i,s in enumerate(SLIDES):
        im=Image.open(OUT/(s['stem']+'.png')).convert('RGB');im.thumbnail((thumbw,thumbh));sheet.paste(im,((i%3)*thumbw,(i//3)*thumbh))
    sheet.save(OUT/'all_slides_contact_sheet.jpg',quality=92)

def verify_and_package():
    import xml.etree.ElementTree as ET
    for filename in ['new_HYP_presentation_15slides.pptx','new_HYP_lecture_zh.docx','new_HYP_statistics.xlsx']:
        with zipfile.ZipFile(OUT/filename) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith(('.xml','.rels')):ET.fromstring(z.read(name))
            if filename.endswith('.pptx'):
                assert len([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)])==15
                assert len([n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n)])==15
    for p,b in SOURCES.items():assert sha(Path(p) if Path(p).is_absolute() else ROOT/p)==b['sha256']
    summary=dict(status='SHOWCASE_ROW_COUNTS_COMPARISONS_SOURCE_HASHES_AND_OPENXML_PASS',
        presentation_only=True,training_updates=0,new_inference=0,bootstrap_recomputed=False,
        pages=15,source_files=SOURCES,script=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha(__file__)),
        cases=CASES,primary='COST1',secondary='CE',scope='GroZi/RPC scoped external; ISIC opened exploratory',
        visual_review='pending rendered-image inspection',plots_not_new_scientific_results=True)
    (OUT/'source_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'slides_manifest.json').write_text(json.dumps(SLIDES,ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('''# new HYP 展示包 · 2026-09-15

- **现场讲解**：打开 `index.html`，方向键翻页，N显示讲解词，F全屏；完全离线。
- **演示文件**：`new_HYP_presentation_15slides.pdf`；`new_HYP_presentation_15slides.pptx`（高清图页，含可编辑演讲者备注）。
- **中文讲义**：`new_HYP_lecture_zh.pdf` 为逐页图文讲解；Word / HTML / Markdown 另含术语、完整表格与FAQ。
- **统计表**：`new_HYP_statistics.xlsx`；`data/*.csv` 可直接导入Excel；包含11模型臂、配对比较、历史模型谱系和选图规则。
- **独立图片**：每页提供 PNG / PDF / SVG；`all_slides_contact_sheet.jpg` 为全部15页缩略总览。
- **真实性**：条形图统计已从原result逐行复算；分组区间复用原封存统计。真实图片只用于事后讲解，不进行新训练或调参。
- **边界**：GroZi/RPC外部确认与ISIC探索分开；COST1主、CE次；旧32/128面板不改写。Ownership、完整过目不忘和新编码器训练为未来工作。
- **复现**：运行 RC/programs/build_rc_new_hyp_showcase_v1.py，需要现有项目Python环境、中文字体与pandoc；原结果与图像来源哈希见source_manifest.json。
''')
    with zipfile.ZipFile(OUT/'new_HYP_showcase_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(OUT))
        z.write(__file__,'build_rc_new_hyp_showcase_v1.py')
    print(json.dumps(dict(status=summary['status'],output=str(OUT),slides=len(SLIDES),sources=len(SOURCES)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    load_results();setup_cases()
    with PdfPages(OUT/'new_HYP_presentation_15slides.pdf') as DECK:build_slides()
    create_xlsx();write_notes();write_web();verify_and_package()
