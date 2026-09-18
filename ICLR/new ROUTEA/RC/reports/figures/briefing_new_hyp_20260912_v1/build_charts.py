"""Read-only experiment audit -> presentation charts. No model or job changes."""
from pathlib import Path
import os
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
import json
import hashlib
import csv
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages

RC = OUT.parents[2]
FONT = '/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'],
    'font.size': 15, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
    'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#CED7E1', 'axes.labelcolor': '#31445B',
    'xtick.color': '#536579', 'ytick.color': '#536579', 'savefig.facecolor': 'white'})
BLUE, TEAL, RED, GRAY, INK, GOLD = '#3275BD', '#148D84', '#C44F58', '#AAB7C7', '#19344F', '#B67B18'
sources = {}
def read(rel):
    p = RC / rel
    b = p.read_bytes()
    sources[rel] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    return json.loads(b)

d90 = read('results/romav2_colnomic_difficult90_frozen_regression_v1/result.json')
e128 = read('results/rc_original7_eval128_full_evidence_v1/result.json')
h593 = read('results/rc_h593_group_risk_strong_base_v1/result.json')
inc = read('results/rc_h593_increment_attribution_v1/result.json')
fixed = read('results/rc_original_mixed96_group_risk_v1/result.json')
for rel in [
    'results/romav2_colnomic_difficult90_frozen_regression_v1/independent_validation.json',
    'results/rc_h593_group_risk_strong_base_v1/result_validation.json',
    'results/rc_h593_increment_attribution_v1/result_validation.json',
    'results/rc_original_mixed96_group_risk_v1/result_validation.json']:
    read(rel)
for rel in ['reports/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md',
            'reports/NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3_20260911.md',
            'reports/REPORT_H593_UNIFIED_HYPOTHESIS_PREDICTION_READOUT_V1_20260911.md',
            'reports/REPORT_NEW_HYP_ORIGINAL7_EVAL128_INDEPENDENT_REVIEW_V1_20260910.md']:
    p = RC / rel
    if p.exists():
        b = p.read_bytes()
        sources[rel] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}

def transition(rows, a, b):
    return (sum(not r['correct'][a] and r['correct'][b] for r in rows),
            sum(r['correct'][a] and not r['correct'][b] for r in rows))
rows = h593['rows']
assert len(rows) == 593 and len({r['query_id'] for r in rows}) == 593
for model, score in h593['scores'].items():
    assert score['correct'] == sum(r['correct'][model] for r in rows)
assert h593['recall_C128'] == sum(r['target_in_C128'] for r in rows) == 570
components = {}
for r in rows:
    components.setdefault(r['component'], set()).add(r['fold'])
assert len(components) == 64 and all(len(v) == 1 for v in components.values())

def record(name, n, base, final, rescue, loss, scope):
    assert final - base == rescue - loss
    return dict(model=name, n=n, raw=base, correct=final, rescue=int(rescue),
                loss=int(loss), net=final-base, accuracy=final/n,
                candidate_source='RAW ColNomic natural C128', scope=scope)

r90 = d90['summaries']['REAL']
r128 = e128['metrics_all128']['REAL']
r32 = fixed['panels']['EVAL32']['rows']
a32, b32 = transition(r32, 'RAW', 'ORIGINAL7')
records = [record('FROZEN_C',90,r90['base_top1'],r90['final_top1'],r90['rescue'],r90['break'],'difficult90 / opened'),
           record('ORIGINAL7',32,sum(r['correct']['RAW'] for r in r32),sum(r['correct']['ORIGINAL7'] for r in r32),a32,b32,'EVAL32 / internal'),
           record('ORIGINAL7',128,r128['RAW_top1'],r128['final_top1'],r128['rescue'],r128['break'],'EVAL128 / internal'),
           record('ALL_COND',593,426,445,*transition(rows,'RAW','ALL_COND'),'593 / grouped OOF5 development'),
           record('GROUP_BASE',593,426,447,*transition(rows,'RAW','GROUP_BASE'),'593 / grouped OOF5 development')]
assert [(r['raw'],r['correct'],r['rescue'],r['loss']) for r in records] == [(61,69,8,0),(25,28,3,0),(88,99,12,1),(426,445,19,0),(426,447,22,1)]

files = []
pdf = PdfPages(OUT/'RouteA_newHYP_briefing_6slides.pdf')
def canvas(title, subtitle, note, source):
    fig = plt.figure(figsize=(16,9), facecolor='white')
    fig.text(.055,.935,title,fontsize=25,weight='bold',color=INK)
    fig.text(.055,.885,subtitle,fontsize=13.5,color='#536579')
    fig.add_artist(plt.Line2D([.055,.945],[.86,.86],transform=fig.transFigure,color='#DFE6EE'))
    fig.text(.055,.07,note,fontsize=11,color='#536579')
    fig.text(.055,.035,'Route A · 数据截至 2026-09-11 | 制图 2026-09-12 | 来源：'+source,fontsize=9,color='#6E7E90')
    return fig
def finish(fig, stem):
    fig.savefig(OUT/f'{stem}.png',dpi=240)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    pdf.savefig(fig)
    files.extend([f'{stem}.png',f'{stem}.pdf',f'{stem}.svg'])
    plt.close(fig)

# 1. Separate panels: not a sample-size scaling curve.
fig = canvas('01  检索收益已经出现，并在更大内部样本上重复',
    '全部来自 RAW 自然 C128；各面板分别比较自己的 RAW，不跨面板拼接或汇总。',
    '593 为五折分别训练的方法成绩；其余为固定头评价。全部为内部／开发证据，不是 untouched 外部确认。',
    'difficult90 result；ORIGINAL7 EVAL128；GROUP_RISK result；fixed-panel result')
titles=['difficult90 · 旧 FROZEN_C','EVAL32 · ORIGINAL7','EVAL128 · 同一 ORIGINAL7','593 图 · 分组五折 OOF']
for i in range(4):
    ax=fig.add_axes([.075+i*.225,.255,.18,.50])
    rec=records[i]
    vals=[rec['raw'],rec['correct']] if i<3 else [426,445,447]
    labels=['RAW','FROZEN_C' if i==0 else 'ORIGINAL7'] if i<3 else ['RAW','ALL_COND','GROUP_BASE']
    colors=[GRAY,BLUE] if i<3 else [GRAY,BLUE,TEAL]
    n=rec['n']
    ax.bar(range(len(vals)),[100*v/n for v in vals],color=colors,width=.62)
    for x,v in enumerate(vals):
        ax.text(x,100*v/n+2,f'{v}/{n}\n{100*v/n:.1f}%',ha='center',fontsize=12,color=INK)
    ax.set_ylim(0,110); ax.set_yticks([0,25,50,75,100]); ax.set_xticks(range(len(vals)),labels,fontsize=10)
    ax.set_title(titles[i],fontsize=13,pad=25,color=INK)
    ax.set_ylabel('Top-1 准确率（%）' if i==0 else '')
    ax.grid(axis='y',color='#E9EEF3'); ax.set_axisbelow(True)
    gain=100*(vals[-1]-vals[0])/n
    gain_label = f'相对本面板 RAW：+{gain:.2f} 个百分点' if i < 3 else f'GROUP_BASE 对 RAW：+{gain:.2f} 个百分点'
    ax.text(.5,-.23,gain_label,transform=ax.transAxes,ha='center',color=TEAL,fontsize=11)
finish(fig,'01_performance_separate_panels')

# 2. Paired counts, not relative error rates.
fig=canvas('02  收益来自真实纠错：同时报告破坏，不只看净增',
    '救回：RAW 错 → 模型对；破坏：RAW 对 → 模型错。各行比较不同样本／参数头，不能相加。',
    '零破坏是这些已观察样本的结果，不是对未知数据的 no-regret 保证。',
    '与图 01 相同；593 计数从逐 query 的 correct 字段重新计算')
ax=fig.add_axes([.30,.24,.57,.54]); yy=np.arange(5)
res=[r['rescue'] for r in records]; losses=[r['loss'] for r in records]
ax.barh(yy,res,color=TEAL,height=.56,label='救回'); ax.barh(yy,[-v for v in losses],color=RED,height=.56,label='破坏')
for y,r in enumerate(records):
    ax.text(r['rescue']+.4,y,str(r['rescue']),va='center',color=TEAL,weight='bold')
    ax.text(-r['loss']-.4,y,str(r['loss']),ha='right',va='center',color=RED,weight='bold')
    ax.text(27,y,f"净增 +{r['net']}",va='center',color=INK,fontsize=14)
ax.set_yticks(yy,['FROZEN_C · difficult90','ORIGINAL7 · EVAL32','ORIGINAL7 · EVAL128','ALL_COND · 593 OOF','GROUP_BASE · 593 OOF'])
ax.invert_yaxis(); ax.set_xlim(-3,31); ax.set_xticks([0,5,10,15,20,25]); ax.set_xlabel('图片数（破坏画在零线左侧）')
ax.axvline(0,color='#7E90A4',lw=1); ax.grid(axis='x',color='#E9EEF3'); ax.set_axisbelow(True)
ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2)
finish(fig,'02_rescue_break_balance')

# 3. Same outcome unit across panels: retention of REAL's rescue set.
rr=e128['REAL_rescue_retention']['CBIND']
real593={r['query_id'] for r in rows if not r['correct']['RAW'] and r['correct']['ALL_COND']}
ret593=sum(r['query_id'] in real593 and r['correct']['ALL_COND_CBIND'] for r in rows)
bind=inc['binding']['CONDITIONAL4']
assert len(real593)==19 and ret593==0 and bind['REAL_rescues']==5 and bind['INCREMENT_BIND_rescues_retained']==0
fig=canvas('03  候选绑定有作用：错配证据会消除原来的纠错',
    '纵轴统一为“REAL 的原救回集合中仍然正确的图片数”，不是控制条件的全部正确数。',
    '整包错绑与新增项错绑是不同干预。它们支持计算绑定依赖，不等于 token／像素级空间因果；第三项为事后探索性归因。',
    'ORIGINAL7 EVAL128；H593 GROUP_RISK rows；H593 INCREMENT_ATTRIBUTION binding')
specs=[('ORIGINAL7 · EVAL128',12,len(rr['retained_query_ids']),'整包候选证据错绑\n总正确数另为 99 → 67'),
       ('ALL_COND · 593 OOF',19,ret593,'整包候选证据错绑\n总正确数另为 445 → 397'),
       ('新增补偿 · 593 OOF',5,0,'冻结 BASE，仅错绑新增项\n5 次新增全部消失；总正确数 445 → 441')]
for i,(name,a,b,caption) in enumerate(specs):
    ax=fig.add_axes([.085+i*.30,.30,.23,.44]); ax.bar([0,1],[a,b],color=[BLUE,RED],width=.56)
    for x,v in enumerate([a,b]): ax.text(x,v+.5,str(v),ha='center',fontsize=23,color=INK)
    ax.set_ylim(0,22); ax.set_yticks([0,5,10,15,20]); ax.set_xticks([0,1],['REAL','错绑后'])
    ax.set_title(name,fontsize=15,pad=20,color=INK); ax.grid(axis='y',color='#E9EEF3'); ax.set_axisbelow(True)
    if i==0: ax.set_ylabel('原救回保留数')
    ax.text(.5,-.24,caption,transform=ax.transAxes,ha='center',fontsize=11,color='#536579',linespacing=1.7)
finish(fig,'03_candidate_binding_controls')

# 4. Strong comparisons retained; no selected control masquerades as main arm.
fig=canvas('04  593 图：组权重修正有效，但额外条件门未稳定胜出',
    '同一 RAW-C128、593 图、64 组、五折；底层证据不变。点图横轴放大显示小差异，不能与图 01 的柱高直接比较。',
    'GROUP_BASE 447 是开发最高正确数之一，不是新的部署头；预定 GROUP_COND 对 ALL_COND 的稳定优势未成立。',
    'H593 GROUP_RISK result / theory_prediction_readout（完整路径见来源清单）')
ax=fig.add_axes([.24,.21,.34,.58])
names=['RAW','ALL_BASE','ALL_CONST','ALL_COND','SMALL_BASE','SMALL_CONST','SMALL_COND','GROUP_BASE','GROUP_CONST','GROUP_COND']
for i,k in enumerate(names):
    v=h593['scores'][k]['correct']; color=TEAL if k=='GROUP_BASE' else (GRAY if k=='RAW' else BLUE)
    ax.scatter(v,i,s=105,color=color); ax.text(v+.45,i,str(v),va='center',color=INK,fontsize=12)
ax.set_yticks(range(len(names)),names,fontsize=12); ax.invert_yaxis(); ax.set_xlim(424,451)
ax.set_xticks([426,430,435,440,445,450]); ax.set_xlabel('正确图片数／593（非零起点）'); ax.grid(axis='x',color='#E9EEF3')
fig.text(.64,.70,'同 ALL 训练行，仅改变组权重',fontsize=16,color=INK,weight='bold')
fig.text(.64,.645,'440 → 447：8 救 / 1 损',fontsize=21,color=TEAL)
fig.text(.64,.59,'组精确检验 p = 0.0078125\n支持该协议中的训练权重效应',fontsize=12,color='#536579',linespacing=1.7)
fig.text(.64,.46,'与更强对照比较',fontsize=16,color=INK,weight='bold')
fig.text(.64,.39,'SMALL_CONST 446 → GROUP_BASE 447\n4 救 / 3 损，稳定优势尚未建立',fontsize=13,color='#536579',linespacing=1.8)
fig.text(.64,.25,'多加条件门并非必需：\nGROUP_BASE 447 → GROUP_COND 446',fontsize=13,color=RED,linespacing=1.8)
finish(fig,'04_training_controls_593')

# 5. Explicitly separate miss and in-set decision failures.
correct=sum(r['correct']['GROUP_BASE'] for r in rows)
miss=sum(not r['target_in_C128'] for r in rows)
inwrong=sum(r['target_in_C128'] and not r['correct']['GROUP_BASE'] for r in rows)
assert (correct,miss,inwrong)==(447,23,123) and not any(not r['target_in_C128'] and r['correct']['GROUP_BASE'] for r in rows)
fig=canvas('05  剩余瓶颈：主要不是 target 没进入候选集',
    'GROUP_BASE · RAW 自然 C128 · 593 图五折开发 OOF；分类从逐图结果重新计算，不推测单一根因。',
    'C128 命中只表示“允许参与竞争”，不表示识别正确。123 个在场错误还需区分内容读取、候选竞争及 HOLD/SWITCH。',
    'rc_h593_group_risk_strong_base_v1/result.json / rows 与 recall_C128')
ax=fig.add_axes([.055,.21,.45,.57]); vals=[correct,inwrong,miss]
ax.pie(vals,colors=[TEAL,RED,GOLD],startangle=90,counterclock=False,wedgeprops={'width':.30,'edgecolor':'white','linewidth':3})
ax.text(0,.08,'447 / 593',ha='center',va='center',fontsize=27,color=INK,weight='bold')
ax.text(0,-.18,f'准确率 {100*correct/593:.2f}%',ha='center',color='#536579',fontsize=15)
for y,label,v,color in [(.71,'最终识别正确',correct,TEAL),(.55,'target 已在 C128，但最终错误',inwrong,RED),(.39,'target 未进入 C128',miss,GOLD)]:
    fig.text(.56,y,label,fontsize=16,color=color,weight='bold')
    fig.text(.56,y-.055,f'{v} 张 / {593} 张（{100*v/593:.2f}%）',fontsize=15,color=INK)
fig.text(.56,.22,f'候选覆盖：570/593 = {100*570/593:.2f}%\n剩余146个错误中，123个（84.2%）target已在场',fontsize=14,color='#536579',linespacing=1.7)
finish(fig,'05_remaining_bottlenecks')

# 6. Diagram: implemented decision HYP is not a spatial localization claim.
fig=canvas('06  new HYP：reference 身份假说，而非已证明的空间分割',
    '现有有效实现：RoMa 的匹配／可见性权重 + ColNomic 的身份内容证据 + 共享决策头。',
    '任务内只使用检索身份监督；骨干预训练另行披露。无身份专属参数，不代表已完成新身份登记与未知分布泛化实验。',
    'NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2；NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3')
ax=fig.add_axes([.055,.17,.89,.64]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',facecolor=color,edgecolor='#D6E1EC',lw=1.2))
    ax.text(x+w/2,y+h*.71,title,ha='center',va='center',fontsize=15,color=INK,weight='bold')
    ax.text(x+w/2,y+h*.31,body,ha='center',va='center',fontsize=11.5,color='#425971',linespacing=1.7)
def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'-|>','color':'#7C91A6','lw':1.7})
box(.01,.61,.20,.28,'RAW 原始检索','全图库 → 自然 C128\n保留原答案 w','#EDF2F8')
box(.27,.61,.30,.28,'每个 reference g 提供证据','RoMa：双端可见性／对应\nColNomic：完整 reference 内容','#E9F4F5')
box(.63,.61,.35,.28,'共享小头：完整候选竞争','127 个 challenger 与 HOLD 比较\n最多一次 SWITCH；否则保持 w','#EAF0FA')
arrow(.216,.75,.263,.75); arrow(.578,.75,.623,.75)
box(.01,.09,.47,.38,'当前机制及内部证据','纠错净增 + 候选级绑定依赖\n联合质量与内容：S = M × L\n共享函数支持 reference 扩展（结构性质）','#EAF5F1')
box(.53,.09,.45,.38,'尚未证明的更强命题','连通区域 HYP ／ 像素级空间 ownership\n独立外部确认 ／ 新身份登记后的可靠性\n当前 logit 不是已校准后验概率','#FAF2E7')
ax.text(.5,.515,'目标：选择“比原答案更可信”的身份解释，而不是先假定哪块区域就是 target。',ha='center',fontsize=12.5,color=INK)
finish(fig,'06_new_hyp_mechanism_and_scope')
pdf.close()

with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
audit={'status':'CHART_DATA_ARITHMETIC_AND_ROW_RECOMPUTATION_PASS',
       'not_new_experimental_validation':True,'date':'2026-09-12','data_cutoff':'2026-09-11',
       'source_files':sources,'performance_records':records,
       'checked_593_unique_queries':593,'checked_components_no_crossfold':64,
       'candidate_recall_593':570,'GROUP_BASE_remaining':{'in_C128_wrong':123,'absent':23},
       'artifacts':files+['RouteA_newHYP_briefing_6slides.pdf','metrics.csv'],
       'model_or_parameter_changes':False,'training_or_slurm_submissions':False}
(OUT/'chart_data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
readme='''# Route A / new HYP 汇报图表（2026-09-12）

数据截至2026-09-11已完成结果。六张16:9图，PNG为3840×2160；另有矢量PDF/SVG和六页合并PDF。
这只是结果可视化，不是新实验，没有修改模型、训练输入、阈值或调度。

## 推荐汇报顺序与讲解

1. `06_new_hyp_mechanism_and_scope`：先交代RAW、reference证据和共享纠错头；new HYP不是空间分割。
2. `01_performance_separate_panels`：展示各自RAW对照下的内部收益，不连成数据规模曲线。
3. `02_rescue_break_balance`：净增由救回减破坏组成，零破坏仅是样本观察。
4. `03_candidate_binding_controls`：错绑消除原救回，支持reference相关证据的作用；不是像素因果。
5. `04_training_controls_593`：同ALL行改组权重440→447，保留所有强对照与条件门负结果。
6. `05_remaining_bottlenecks`：23缺席、123在场仍错，后者不能简单归于缺乏图像或同一个根因。

## 固定口径

- RAW=原始ColNomic检索；所有图的候选源是RAW自然C128，没有把D1收益混入。
- FROZEN_C与ORIGINAL7不是同一参数头。旧FROZEN_C EVAL32为25→27；图01的EVAL32使用ORIGINAL7，因此25→28。
- 32/90/128/593是query图片数；C128是每张query的候选数。不同面板可能有历史重用，绝不相加。
- 593是开发复用的分组五折，每折重新训练，不是一个最终全训练部署头；不是独立外部确认。
- 图04点图使用非零横轴并明确标注；图01准确率柱状图从0开始。没有构造未经验证的误差棒。
- 19/0、22/1分别属于ALL_COND与GROUP_BASE，不是一个模型的两种计数。
- 所有原始negative/control结果保留；这些图没有宣称严格无损、空间ownership、普遍最优或未经测试的定位能力。

## 可追溯与复现

`metrics.csv`提供性能数值，`chart_data_audit.json`列出来源文件SHA256及计数复核。
`build_charts.py`直接读取原JSON，重新核对593逐图正确数、救回/破坏、候选缺席、组划分及冻结32图。
此审计不替代原实验的独立validator；bootstrap未重新计算；图中p值引用已封存报告。
中文字体使用系统Droid Sans Fallback，程序将Matplotlib缓存留在本输出目录，不修改home配置。

复现：`python3 build_charts.py`。合并PDF：`RouteA_newHYP_briefing_6slides.pdf`。
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
with zipfile.ZipFile(OUT/'RouteA_newHYP_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.suffix in {'.png','.svg','.pdf','.csv','.json','.md','.py'}:
            z.write(p,p.name)
print(json.dumps({'status':audit['status'],'output':str(OUT),'charts':6,'png_size':[3840,2160]},ensure_ascii=False))
