#!/usr/bin/env python3
"""Post-result synthesis only; no fitting, selection or new private data."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_six_cause_isolation_v1'

def read(p):return json.loads(Path(p).read_text())
def binding(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def check_result(part):
    p=OUT/part/'result.json';v=read(OUT/part/'validation.json');assert v['result']==binding(p)
    return read(p)
def contrast(rows,new,old):
    by=defaultdict(list)
    for r in rows:by[r['component']].append(int(r['correct'][new])-int(r['correct'][old]))
    d=np.array([np.mean(v) for _,v in sorted(by.items())]);idx=np.random.default_rng(20260911).integers(0,len(d),(10000,len(d)));ci=np.quantile(d[idx].mean(1),[.025,.975]).tolist()
    return dict(rescue=sum(r['correct'][new] and not r['correct'][old] for r in rows),loss=sum(not r['correct'][new] and r['correct'][old] for r in rows),net=sum(int(r['correct'][new])-int(r['correct'][old]) for r in rows),equal_component_difference=float(d.mean()),component_bootstrap95=ci)

def main():
    inv=check_result('inventory');b=check_result('bridge');c=check_result('coverage');l=check_result('loss_binding');vis=read(OUT/'visual/visual_review.json')
    assert [c['counts'][m] for m in ['RAW','ALL_COST4','ALL_CE']]==[426,440,486]
    assert [l['counts'][m] for m in ['RAW','ALL_COST4','GROUP_BASE','COST1','ALL_CE','RAW2_CE','ALL_CE_CBIND']]==[426,440,447,481,486,426,281]
    for part,total,glob in [('bridge',24,'fit*/validation.json'),('coverage',35,'fit*/validation.json'),('loss_binding',5,'fold*/validation.json')]:
        validations=list((OUT/part).glob(glob));assert len(validations)==total
        for p in validations:
            v=read(p);assert v['payload']==binding(v['payload']['path'])
    pairs=[('COST1','ALL_COST4'),('ALL_CE','COST1'),('COST1','GROUP_BASE'),('ALL_CE','GROUP_BASE'),('COST1','RAW'),('ALL_CE','RAW'),('ALL_CE','RAW2_CE'),('ALL_CE_CBIND','ALL_CE')]
    comparisons={old+'__to__'+new:contrast(l['rows'],new,old) for new,old in pairs}
    for key,v in comparisons.items():
        if key in l['comparisons']:
            ref=l['comparisons'][key];assert v['rescue']==ref['rescue'] and v['loss']==ref['loss'] and np.allclose(v['component_bootstrap95'],ref['component_bootstrap95'],rtol=0,atol=1e-14)
    folded={str(f):{m:sum(r['correct'][m] for r in l['rows'] if r['fold']==f) for m in ['RAW','ALL_COST4','GROUP_BASE','COST1','ALL_CE','RAW2_CE','ALL_CE_CBIND']} for f in range(5)}
    trainheld={}
    for rep in ['S6','FREE','POOL','MATCH','CHANNEL','CHANNEL_PERM']:
        m=rep+'_LINEAR_CE';fit=[p['models'][m] for p in b['fit_diagnostics'] if m in p['models']];n=sum(p['train_queries'] for p in fit)
        trainheld[rep]=dict(train_correct=sum(p['train_correct'] for p in fit),train_occurrences=n,train_mean_CE=sum(p['final_CE']*p['train_queries'] for p in fit)/n,held_correct=b['counts'][m],held_mean_CE=np.mean([r['held_ce'][m] for r in b['rows']]),parameter_count=fit[0]['parameter_count'])
    ce_models=[m for m in c['counts'] if '_CE' in m]
    assert all(c['counts'][m]>c['counts'][m.replace('_CE','_COST4')] for m in ce_models)
    six=[
      dict(priority=1,cause='照片是否包含身份信息',status='部分排除，逐图仍有未决',supported='593中没有像素/标签精确冲突；原TRAIN20错例中至少3例有直接可见区分文字。',unresolved='不能证明每图可识别；近似背面、遮挡和低清晰度仍需逐图证据。'),
      dict(priority=2,cause='冻结tokens是否保留可用信息',status='足以解释本轮提升；全部细节可读性未证明',supported='相同冻结tokens与原六统计已支持H593 486，440停滞并非该表示的硬上限。',unresolved='无精确token碰撞不证明所有细粒度文字被编码；高维头训练全对也不证明身份泛化。'),
      dict(priority=3,cause='MaxSim、聚合和接口压缩',status='当前有限扩展未带来总体优势',supported='保留原六统计，增加FREE/POOL/MATCH/CHANNEL得到113/112/110/74，原S6为114；CHANNEL求和精确重建FREE。',unresolved='不能由这些探针失败推出所有压缩无损；也不能把单例救回当作群体机制成功。'),
      dict(priority=4,cause='表达能力、成本与训练目标',status='H593主要可修失效环节已定位',supported='COST4→COST1 440→481、44救3损；COST1→CE 481→486、14救9损，后一步组区间跨0。CE的57次救回全部已是旧头最强challenger，被HOLD拒绝。',unresolved='这是当前固定协议下的因果对照，不证明所有COST4优化器都无法提升，也未解释旧固定EVAL全部错误。'),
      dict(priority=5,cause='照片数量、身份覆盖和分布',status='已区分本轮作用，不能归为单纯缺数据',supported='等128预算下CE broad482/484/486，COST4 broad449/450/439；扩大身份范围净0/1/4不稳定。每身份1→2照片CE净1/10/18，2→4净2/4/-2。',unresolved='采样共享同一开发人口；更远域和新来源效果未确认。'),
      dict(priority=6,cause='稳定性、重复开发和独立确认',status='内部支持增强，外部确认未完成',supported='CE相对COST4五折全部净增；三次预定采样的方向支持目标作用；对447强基线49救10损，组区间为正。',unresolved='593全部已开发，剩余未触碰可用图为0；需新独立身份/来源确认，不能靠重分593证明外部泛化。')]
    remaining=dict(total_errors=593-l['counts']['ALL_CE'],target_absent_errors=sum(not r['target_in_C128'] for r in l['rows']),candidate_present_errors=sum(r['target_in_C128'] and not r['correct']['ALL_CE'] for r in l['rows']))
    assert remaining==dict(total_errors=107,target_absent_errors=23,candidate_present_errors=84)
    result=dict(status='SIX_CAUSE_BOUNDED_ANALYSIS_COMPLETE',sources={p:binding(OUT/p/'result.json') for p in ['inventory','bridge','coverage','loss_binding']},visual_source=binding(OUT/'visual/visual_review.json'),six=six,counts=l['counts'],comparisons=comparisons,fold_counts=folded,representation_train_held=trainheld,action_diagnosis=l['action_diagnosis'],all_six_assessed=True,all_scientific_uncertainties_resolved=False,new_HYP_external_GO=False,original_fixed_EVAL_retested=False,remaining_errors=remaining,all16_sampling_regimes_CE_gt_COST4=len(ce_models),limits=['COST1/RAW2/binding followup was declared after observing486','No universal label-free or annotation-free pretraining claim; only retrieval identity supervision','Group bootstrap describes this opened cohort; overlapping folds/seeds are not independent external studies'])
    write(OUT/'analysis.json',result)
    fig,axes=plt.subplots(2,2,figsize=(13.5,9));fig.suptitle('Six-cause diagnosis | previously opened development data',fontsize=16)
    names=['RAW','COST4','GROUP','COST1','CE','RAW-only CE','CE + CBIND'];keys=['RAW','ALL_COST4','GROUP_BASE','COST1','ALL_CE','RAW2_CE','ALL_CE_CBIND'];values=[l['counts'][k] for k in keys]
    ax=axes[0,0];bars=ax.bar(range(7),values,color=['#8c9aa5','#e69f00','#999999','#009e73','#0072b2','#8c9aa5','#cc6677']);ax.bar_label(bars,padding=3);ax.set_xticks(range(7),names,rotation=25,ha='right');ax.set_ylim(0,600);ax.set_ylabel('Correct / 593');ax.set_title('Same H593 folds: decision objective and binding')
    ax=axes[0,1];reps=list(trainheld);x=np.arange(len(reps));ax.plot(x,[100*trainheld[k]['train_correct']/trainheld[k]['train_occurrences'] for k in reps],'o-',label='TRAIN occurrences (overlap across folds)',color='#e69f00');ax.plot(x,[100*trainheld[k]['held_correct']/128 for k in reps],'o-',label='Held queries: original TRAIN128 OOF',color='#0072b2');ax.set_xticks(x,['6 stats','+ free','+ pooled dist.','+ pair dist.','+ channels','+ perm. ch.'],rotation=25,ha='right');ax.set_ylim(50,104);ax.set_ylabel('Accuracy (%)');ax.set_title('More expressive inputs can overfit');ax.legend(fontsize=8)
    ax=axes[1,0]
    for mode,color,mark in [('COST4','#e69f00','s'),('CE','#0072b2','o')]:
        for subset,style in [('BROAD128','-'),('NARROW128','--')]:ax.plot([0,1,2],[c['counts'][f'{subset}_{mode}_S{s}'] for s in range(3)],marker=mark,linestyle=style,color=color,label=mode+' / '+('broad' if subset=='BROAD128' else 'narrow'))
    ax.set_xticks([0,1,2]);ax.set_xlabel('Predeclared training-sample draw');ax.set_ylabel('Held correct / 593');ax.set_ylim(420,500);ax.set_title('Exactly 128 training images per fold');ax.legend(fontsize=8,ncol=2)
    ax=axes[1,1]
    for s,color in enumerate(['#0072b2','#009e73','#cc79a7']):ax.plot([1,2,4],[c['counts'][f'PER_ID_{k}_CE_S{s}'] for k in [1,2,4]],'o-',label=f'Draw {s}',color=color)
    ax.set_xticks([1,2,4]);ax.set_xlabel('Images per fixed training identity');ax.set_ylabel('Held correct / 593');ax.set_ylim(460,495);ax.set_title('Fixed identity set, nested images, CE');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.text(.5,.012,'H593 uses 5 source/identity-component folds. TRAIN128 uses its original 4 folds. Neither is untouched external confirmation.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(OUT/'six_cause_summary.png',dpi=180);fig.savefig(OUT/'six_cause_summary.pdf');plt.close(fig)
    lines=['# 六项原因隔离完成：主要可修瓶颈在动作训练成本','','## 先看同一人口、同一协议的结果','','H593：593张、68身份、64个identity/source component，原五折不变；冻结RAW全gallery自然C128、原六统计和七参数头、FP64、2000步末点。RAW/HOLD为零logit，完整127 challenger最大logit>0才SWITCH。所有593留出图保留，包含23张target未进入C128。已开放开发人口；不是旧固定EVAL32/128，也不是外部确认。','','| 模型/训练目标 | 正确/593 | 对原COST4的含义 |','|---|---:|---|','| RAW | 426 | 原始检索 |','| 原七参数 COST4 | 440 | 完整保留旧参数/预测 |','| 既有 GROUP_BASE | 447 | 历史强基线，分开核对 |','| 同七参数 COST1 | 481 | 44救3损，净+41 |','| 同七参数 全C128 CE | 486 | 57救11损，净+46 |','| RAW差+bias两参数 CE | 426 | 单独RAW校准未复现收益 |','| 冻结CE头 + CBIND输入 | 281 | 正确候选绑定被破坏后显著退化 |','','**最重要的失效定位：CE救回的57张，旧COST4已经把target放在最强challenger位置，57张全部被HOLD拒绝。11张损失全部来自原本正确的RAW/HOLD被错误切换。** 本轮证据首先支持决策成本/选择行为失配，不支持“旧表示只能做到440”。阈值保持0，没有通过调阈值制造提升。','','## 成本与全候选目标已分开','','COST4→COST1保持同数据、原六统计、七参数、优化器、步数和原FULL风险形式，只把错误切换惩罚4改为1。COST1→CE再改全候选目标。因此不能把46净增都归因于CE：成本改动自身取得41净增；CE相对COST1再14救9损、净+5，其等component区间跨0。','','CE对GROUP_BASE447为49救10损、净+39，等component差约+8.20pp，bootstrap95%约[+3.61,+13.11]pp。COST1与RAW/强基线的独立配对计数见analysis.json。组区间描述本开发人口，不把共享训练的折和重采样当作新的独立试验。','','原风险在target为challenger时为 `softplus(-z_target) + 4*softplus(max_wrong_challenger z)`；COST1去掉4。CE为 `logsumexp([0,z_1,...,z_127]) - z_target`。这说明retrieval-only定义了监督来源，训练损失仍须与评价所需的净准确率相匹配。没有新的人工框、mask、位置或图片类型监督。','','## 六项逐项结论','','| 优先度 | 问题 | 当前状态 | 已支持的结论 | 仍不能声称 |','|---:|---|---|---|---|']
    for s in six:lines.append(f"| {s['priority']} | {s['cause']} | {s['status']} | {s['supported']} | {s['unresolved']} |")
    lines+=['','## 信息保留与过拟合的直接对照','','原TRAIN128的原四折，所有新臂保留BASE7及其六统计，增加的分量为残差输入；不存在强迫丢掉原功能的接口。CHANNEL的128分量求和以≤2e-12复原FREE，全部16缓存Torch/NumPy重算通过。连续帽函数分布使用16区间17节点，POOL读取MaxSim后query-token分布，MATCH读取完整余弦矩阵分布；两者是有限探针。','','| 输入（线性CE） | 参数数 | TRAIN正确/重叠出现次数 | 留出正确/128 | TRAIN CE | 留出CE |','|---|---:|---:|---:|---:|---:|']
    for rep,d in trainheld.items():lines.append(f"| {rep} | {d['parameter_count']} | {d['train_correct']}/{d['train_occurrences']} | {d['held_correct']} | {d['train_mean_CE']:.4f} | {d['held_mean_CE']:.4f} |")
    lines+=['','真实与错配CHANNEL均能把训练图全部拟合，留出却分别74与84，不能把训练全对称为token身份机制成功。这是明确的过拟合表现，不证明所有更丰富读取器都失败，也不证明每一类细粒度线索已进入token。原S6为114；逐分量二次基函数和UNIT1交叉对照完整计数见bridge/result.json，未超过114。','','## 数量、身份覆盖及稳定性','','同为128训练图，broad覆盖每折52–56身份，narrow约26–28身份；CE broad为482/484/486，narrow482/483/482。扩大身份集合的总体净增0/1/4，等组方向并非三次一致，因此不能称身份覆盖是主要根因。原COST4 broad只有449/450/439；相同采样而更换目标，影响明显大于本范围的数量/覆盖变化。','','固定每折42–46身份，每身份1/2/4张分别约42–46、84–92、168–184训练图；CE三次曲线为480→481→483、470→480→484、469→487→485。1→2有帮助，2→4已不稳定。ALL每折451–461个recall-present训练query得到486。这里不是越多图片必然越好，更不是“完整593各折都当训练数据”。','','## 理论解释与下一步边界','','目前有证据的统一解释是：正确绑定的reference内容/匹配质量提供相对身份证据；一个检索标签监督的小型决策头，必须按目标任务的错误成本学习是否推翻RAW。原系统确实读到了许多正确挑战者，保守风险使它拒绝行动。调整任务成本保留了retrieval-only、相同信息接口、相同零阈值和候选绑定，并在当前H593协议取得群体净增。','','这可以支持new HYP的“reference条件化证据与任务一致的身份决策”解释；并不是一个已证明普遍成立的新定理，也不是空间ownership。旧固定28/32、99/128本轮没有重测；以前CE在mixed PAIR64+FULL32上仅101/128且26/32，不能用这次H593486覆盖那条历史记录。','','当前107个CE错误分为23个候选缺失、84个候选内错误。前者不可能仅靠C128内action修复。库存审计为987−正式排除392=595记录、593唯一图，全部已开发；没有可直接当作新确认集的剩余图。COST1/CE配方与全部折头已经封存；独立确认须新增身份/来源，不能用再次划分593替代。','','所有六项已执行可在当前授权数据上完成的有限检验，但“每张图/token是否充分”和“外部泛化”仍未得到普遍证明。没有部署替换、原面板重测或HYP external GO。','','## 物证与验证','','- inventory/result.json、visual/visual_review.json：库存与20错例；','- bridge/result.json、bridge/validation.json：16缓存、24输入×折训练；','- coverage/result.json、coverage/validation.json：35数量/覆盖/目标分片；','- loss_binding/result.json、loss_binding/validation.json：5折成本与绑定补充；','- analysis.json：独立配对重算、逐折结果、六项状态；','- six_cause_summary.png / six_cause_summary.pdf：可分享图。','','![六项结果](../results/rc_six_cause_isolation_v1/six_cause_summary.png)','']
    report=ROOT/'reports/REPORT_SIX_CAUSE_ISOLATION_V1_20260913.md'
    with report.open('x') as f:f.write('\n'.join(lines))
    write(OUT/'analysis_validation.json',dict(status='SIX_CAUSE_SYNTHESIS_SOURCE_HASH_AND_PAIRED_COUNT_PASS',analysis=binding(OUT/'analysis.json'),report=binding(report),plot=binding(OUT/'six_cause_summary.png'),original_fixed_EVAL_reads=0,paired_comparisons=8,all_six_assessed=True,all_scientific_uncertainties_resolved=False))
    print(json.dumps(dict(counts=l['counts'],comparisons=comparisons,remaining_errors=remaining,report=str(report)),ensure_ascii=False))

if __name__=='__main__':main()
