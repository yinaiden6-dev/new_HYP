#!/usr/bin/env python3
"""Localize existing A/J/P interventions without any new model execution.

Full factorial direct-input contrasts use the fixed original per-fold COST1
head on the common41 panel. Fixed-original and reselected wrong competitors
are separated. Visual55 is kept as an independent, larger cached panel.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import numpy as np
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_pair_relation_localization_v1/inside_paths'
SUBSET = ROOT/'results/rc_h593_subset41_frozen_paths_v1'
INSIDE = ROOT/'results/rc_h593_m_inside_v1'
VISUAL55 = ROOT/'results/rc_h593_visual55_mechanism_v1'
FACTORS = tuple('A%dJ%dP%d' % x for x in itertools.product((0,1), repeat=3))
MODES = ('M_ONLY','FULL_UV')
STAGES = ('COARSE','HR1')
VISUAL_ARMS = ('Q_GRAY','R_GRAY','Q_LOWPASS','R_LOWPASS','Q_SHUFFLE','R_SHUFFLE')
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_INDICES = {}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(binding):
    assert bind(binding['path']) == binding, binding['path']
    return Path(binding['path'])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def avg(values):
    return statistics.mean(values) if values else None


def grouped(records, values):
    pairs = [(r['component'],v) for r,v in zip(records,values) if v is not None]
    if not pairs:return dict(queries=0,groups=0,query_mean=None,group_equal_mean=None)
    groups = sorted({g for g,v in pairs})
    means = [avg([v for gg,v in pairs if gg == g]) for g in groups]
    if len(means) not in BOOTSTRAP_INDICES:
        BOOTSTRAP_INDICES[len(means)] = np.random.default_rng(BOOTSTRAP_SEED).integers(
            0,len(means),size=(BOOTSTRAP_DRAWS,len(means)))
    boot = np.asarray(means)[BOOTSTRAP_INDICES[len(means)]].mean(axis=1)
    return dict(queries=len(pairs), groups=len(groups), query_mean=avg([v for g,v in pairs]),
                group_equal_mean=avg(means), positive_groups=sum(v>0 for v in means),
                negative_groups=sum(v<0 for v in means),
                positive_queries=sum(v>0 for g,v in pairs), negative_queries=sum(v<0 for g,v in pairs),
                zero_queries=sum(v==0 for g,v in pairs),
                group_bootstrap_percentile95=np.quantile(boot,[.025,.975]).tolist(),
                leave_one_group_out_range=([min((sum(means)-v)/(len(means)-1) for v in means),
                    max((sum(means)-v)/(len(means)-1) for v in means)] if len(means)>1 else None))


def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); out=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):out[order[k]]=(i+j-1)/2+1
        i=j
    return out


def spearman(x,y):
    a=np.asarray(ranks(x));b=np.asarray(ranks(y));a-=a.mean();b-=b.mean()
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/den) if den else None


def component_diagnostics(inside, target, head_wrong, mass_wrong):
    candidates=[];full=[]
    for record,source in zip(inside['records'],inside['sources']):
        data=torch.load(checked(source),map_location='cpu',weights_only=True)
        assert data['authority']==inside['authority'] and set(data['sides'])=={0,1}
        assert data['reference_sha']==Path(source['path']).stem
        sides=[]
        for side in (0,1):
            cc=data['sides'][side]['components']; assert cc['matching_position']['shape'][-1]==1024
            sides.append({name:float(cc[name]['rms'])for name in ('appearance','joint_context','matching_position')})
        cq,cr=[2*s['matching_position']**2 for s in sides]
        assert 0<=cq<=1.00001 and 0<=cr<=1.00001
        m=record['branches']; logm={k:math.log(m[k]['HR1'])for k in FACTORS}
        candidates.append(dict(physical_row=record['physical_row'],candidate_position=record['candidate_position'],
            side_rms=sides,P_concentration_Q=cq,P_concentration_R=cr,
            P_concentration_bidirectional_mean=(cq+cr)/2,
            P_concentration_bidirectional_geometric_mean=math.sqrt(cq*cr),
            P_to_J_rms_geometric_mean=math.sqrt(np.prod([s['matching_position']/s['joint_context']for s in sides])),
            P_to_A_rms_geometric_mean=math.sqrt(np.prod([s['matching_position']/s['appearance']for s in sides])),
            J_rms_geometric_mean=math.sqrt(sides[0]['joint_context']*sides[1]['joint_context']),
            A_rms_geometric_mean=math.sqrt(sides[0]['appearance']*sides[1]['appearance']),
            M_native_HR1=m['A1J1P1']['HR1'],logM_P_effect_J1=logm['A1J1P1']-logm['A1J1P0'],
            logM_P_effect_J0=logm['A1J0P1']-logm['A1J0P0'],
            logM_JP_interaction=logm['A1J1P1']-logm['A1J1P0']-logm['A1J0P1']+logm['A1J0P0']))
        if data['full_first_pair_components']:
            assert all(data['sides'][s]['first_pair_components'] is not None for s in (0,1))
            full.append(source)
        else:assert all(data['sides'][s]['first_pair_components'] is None for s in (0,1))
    metrics={}
    cs=[c['P_concentration_bidirectional_mean']for c in candidates]
    for xfield in ('P_concentration_bidirectional_mean','P_concentration_bidirectional_geometric_mean',
                   'P_to_J_rms_geometric_mean','P_to_A_rms_geometric_mean','J_rms_geometric_mean','A_rms_geometric_mean'):
        xx=[c[xfield]for c in candidates]
        for yfield in ('M_native_HR1','logM_P_effect_J1','logM_P_effect_J0','logM_JP_interaction'):
            metrics['rho_'+xfield+'_vs_'+yfield]=spearman(xx,[c[yfield]for c in candidates])
    if target is not None:
        cwrong=max((i for i in range(128)if i!=target),key=lambda i:cs[i])
        metrics.update(target_C_rank_desc=129-ranks(cs)[target],
            target_M_rank_desc=129-ranks([c['M_native_HR1']for c in candidates])[target],
            target_C_logratio_vs_original_M_wrong=math.log(cs[target]/cs[mass_wrong]),
            target_C_logratio_vs_original_head_wrong=math.log(cs[target]/cs[head_wrong]),
            target_C_logratio_vs_new_C_wrong=math.log(cs[target]/cs[cwrong]))
    return dict(candidates=candidates,metrics=metrics,full_tensor_sources=full)


def contrasts(v):
    if any(v[k] is None for k in FACTORS):
        return None
    jp1 = v['A1J1P1']-v['A1J1P0']-v['A1J0P1']+v['A1J0P0']
    jp0 = v['A0J1P1']-v['A0J1P0']-v['A0J0P1']+v['A0J0P0']
    return dict(P_given_A1J1=v['A1J1P1']-v['A1J1P0'],
                P_given_A1J0=v['A1J0P1']-v['A1J0P0'],
                J_given_A1P1=v['A1J1P1']-v['A1J0P1'],
                J_given_A1P0=v['A1J1P0']-v['A1J0P0'],
                A_given_J1P1=v['A1J1P1']-v['A0J1P1'],
                JP_given_A1=jp1, JP_given_A0=jp0, AJP_difference=jp1-jp0)


def head_path(mode, factor, stage):
    return mode+('/native/' if factor=='A1J1P1' else '/inside/'+factor+'/')+stage


def make_report(out,rescues):
    def stat(x):
        interval=x['group_bootstrap_percentile95']
        return f"{x['group_equal_mean']:+.6f} [{interval[0]:+.6f}, {interval[1]:+.6f}]；正效应 {x['positive_queries']}/{x['queries']} 图、{x['positive_groups']}/{x['groups']} 组"
    lines=['# 既有缓存：A/J/P 内部路径的新增定位分析', '',
        '范围：已打开的 H593 common41 开发面板，原每折七参数 COST1 头、原 natural C128、原 HOLD/SWITCH 门（HOLD logit=0），不重新训练。RAW 32/41，原完整路径 35/41，3 个原纠错。目标在候选中的 38 图 / 28 个原 component 组用于目标对错误候选 margin；全部41图 / 30组用于整体候选统计。视觉55另有52个目标在候选的图 / 31组，不混入41图估计。', '',
        '本次新增的是对既有所有世界的成对差分、固定竞争者分解、J×P 交互、P 范数集中度和视觉变换配对比较；没有新模型前向、GPU 作业或重训。', '',
        '## 路径含义与可解释边界', '',
        'A 是直接传入预测器的外观特征；J 是跨图联合上下文特征。P 不是一个预测坐标，而是原生 J 两侧余弦相似度经温度0.1 softmax后，对另一幅图坐标的1024维正余弦傅里叶特征求加权平均。P 由原生 J 派生，因此 J/P 的直接输入置零不是两份独立信息的删除。A-off 同样保留从原 A 派生出的 J/P。', '',
        '所有分支保持原生 warp 与后续 refiner 证据。M_ONLY 只替换标量 M，保留原空间加权读出；FULL_UV 同时替换 M 和对应的 query/reference 空间权重，再通过同一原头评分。P-off 不会移除 J 的空间排列、原生 warp 等其他位置通路。最终 logM/head margin 的交互还包括 sigmoid、聚合及头特征的非线性，不能全部归入上游特征的独立协同。置零同时改变输入能量与分布，这不是保持范数只扰乱坐标结构的干预。', '',
        '固定错误对手：每图原 FULL_UV/native/HR1 头中得分最高的错误物理候选，包含 RAW/HOLD，随后所有世界固定该候选。目标对重新选择的最强错误对手另存，避免把竞争者替换混作固定成对效应。', '',
        '## J/P 联合作用', '',
        '以下为 HR1 目标减固定原错误对手 margin，组等权平均；方括号为20,000次 component 组 bootstrap 95% percentile区间，seed=20260924。图/组的正效应计数均直接计数，没有显著性阈值。', '',
        '| 固定原头读出 | 加入P，保留J | 加入P，去掉J | J×P = F111−F110−F101+F100 |',
        '|---|---|---|---|']
    for mode in MODES:
        c=out['factorial_summary'][mode+'/HR1']['contrasts']['target_vs_fixed_original_wrong']
        lines.append('| '+mode+' | '+' | '.join(stat(c[k])for k in ('P_given_A1J1','P_given_A1J0','JP_given_A1'))+' |')
    lines += ['', 'M_ONLY 的 J×P 描述性区间为正；FULL_UV 的 J×P 区间跨0，两种读出中 P|J1 的平均效应区间也均跨0。因此不能将这些组均值上升为普遍稳定的正协同；下面的具体决策依赖不依赖总体显著性判断。', '',
        '原纠错保留与总正确数（HR1；后者包含可能新增与损失，不能当作原纠错保留数）：', '',
        '| A/J/P 世界 | M_ONLY 正确 / 41；原3纠错保留 | FULL_UV 正确 / 41；原3纠错保留 |',
        '|---|---|---|']
    for factor in FACTORS:
        cells=[]
        for mode in MODES:
            w=out['factorial_summary'][mode+'/HR1']['worlds'][factor]
            cells.append(f"{w['correct']}/41；{w['original3_rescues_retained']}/3")
        lines.append('| '+factor+' | '+' | '.join(cells)+' |')
    lines += ['', '逐个原纠错的目标相对 HOLD logit（正值才可跨越 HOLD，最终还需胜过其他错误候选）：', '',
        '| 原query | 路径 | 完整111 | P-off 110 | J-off 101 | J/P-off 100 |',
        '|---|---|---:|---:|---:|---:|']
    for r in rescues:
        for mode in MODES:
            w=r['factorial'][mode+'/HR1']['worlds']
            lines.append('| '+r['original_query_id']+' | '+mode+' | '+' | '.join(f"{w[k]['target_logit']:+.6f}"for k in ('A1J1P1','A1J1P0','A1J0P1','A1J0P0'))+' |')
    lines += ['', '这定位到一个可复核的条件因果链：保持 J，只删除显式 P 输入即可使部分原纠错跨回 HOLD；FULL_UV 与 M_ONLY 的差异进一步定位到空间权重读出。它支持 P 通路有实际决策贡献，不证明 P 所携带的是正确物理对应，也不证明保留下来的决策不使用几何。', '',
        '同一38图/28组上，加入 J、保留 P 时：', '']
    mass=out['mass_factorial_summary']['HR1']
    for field,label in [('candidate_mean_logM_target_present_only','候选平均 logM'),
                        ('target_vs_original_M_wrong_logratio','目标相对固定原最高M错误候选的 logM 比')]:
        lines.append('- '+label+'：'+stat(mass[field]['J_given_A1P1'])+'。')
    lines += ['', '因此 J 的作用不能概括为普遍抬高质量值；它对候选整体作抑制，同时在该面板提高目标的相对选择性。该描述是已有分支的计算效应，不识别正确物理匹配或全局布局一致性。', '',
        '## P 的范数包含什么', '',
        '设 P_i=Σ_j a_ij[sin(Ωx_j),cos(Ωx_j)]，Ω有K=512个频率，则 ||P_i||²/K=Σ_jk a_ij a_ik (1/K)Σ_l cos(ω_l·(x_j−x_k))。每侧全位置平均量恰为 2*rms(P)²。它是匹配分布的傅里叶核集中度；不是坐标均值、熵或几何真值。错误但集中的对应也能有较高值，有限频率核也不等于通用单调空间距离。', '',
        '预先固定主诊断 C=(C_Q+C_R)/2；双向几何平均和P/J、P/A RMS比仅作辅助。以下 Spearman 先在每图128候选内计算，再组等权汇总，避免把不同图的基准值和数千候选当成独立样本。', '',
        '| C 的图内候选相关性 | 组均值 [95%区间]；正效应图/组 |', '|---|---|']
    cc=out['concentration_summary']
    for field,label in [('M_native_HR1','原生 M'),('logM_P_effect_J1','保留J时加入P的logM效应'),
                        ('logM_P_effect_J0','去掉J时加入P的logM效应'),('logM_JP_interaction','J×P logM交互')]:
        lines.append('| '+label+' | '+stat(cc['rho_P_concentration_bidirectional_mean_vs_'+field])+' |')
    counts=cc['rank_counts']
    lines += ['',f"38个目标在候选的图：C 目标top1={counts['target_C_top1']}/38；原M目标top1={counts['target_M_top1']}/38；目标C高于原最高M错误候选={counts['target_C_gt_original_M_wrong']}/38。没有拟合、选择阈值或将C接入原头。", '',
        '这些相关性只检查 P 的强度/集中度与质量及干预效应的联系；P置零的扰动幅度本身随||P||增大，相关性可能部分来自输入扰动幅度。因此高相关不能说明原预测器专门读取集中度或只读取范数，也不能证明 P 对决策的贡献已经被范数完全解释。完整相位、位置依赖和跨位置一致性是否另有贡献仍不可由这些标量汇总识别。', '',
        '## 视觉变换的新增成对比较', '',
        '独立 visual55 的52图/31组，目标对每世界最强错误候选的 margin 差。负号表示前一种变换比后一种变换更削弱目标的竞争余量。', '',
        '| 变换差 | M_ONLY | FULL_UV |', '|---|---|---|']
    for name,x in out['visual55_paired_comparisons'].items():
        lines.append('| '+name+' | '+stat(x['summary']['M_ONLY_margin_difference'])+' | '+stat(x['summary']['FULL_UV_margin_difference'])+' |')
    lines += ['', '灰度、低通和分块打乱的强度与影响不正交：低通删除细节，打乱同时改变布局、边缘接缝与上下文。当前没有这些变换与 P 分支的联合干预，不能把变换损伤直接归因为 P，也不能把所有空间效应称为几何对应验证。', '',
        '## 可复核性与缓存可用性', '',
        f"逐一 SHA 验证 {out['tensor_readiness']['verified_pair_files']} 个内部 pair 文件；每个有双向 A/J/P RMS 与16×16空间汇总。完整 A/J/P 张量仅有 {len(out['tensor_readiness']['complete_AJP_pair_files'])} 个 pair（query000首pair）。除该pair外，不能从这些汇总直接重放保范数P置换。", '',
        '读取41图×85路径×127个已保存logit并复核动作/正确性/目标margin；这不是新运行上游模型。源文件路径和SHA、逐图世界、逐候选集中度以及数值一致性检查见 summary.json / rows41.json / validation.json。未读取 protected formal392 或 D1-MI。', '',
        'bootstrap仅刻画已打开面板按原component组重抽样的描述性不确定性，没有多重比较校正，不能替代外部确认或推广为所有检索任务的机制定理。', '']
    return '\n'.join(lines)


def main():
    opened = read(SUBSET/'result.json')
    auth_path = ROOT/'registry/rc_h593_subset41_frozen_paths_authority_v1_20260923.json'
    authority = read(auth_path)
    old = {r['query_id']:r for r in read(checked(authority['old_result']))['rows']}
    gallery = {r['physical_row']:r['identity'] for r in read(checked(authority['gallery']))['records']}
    sources = [bind(__file__), bind(SUBSET/'result.json'), bind(auth_path), authority['old_result'], authority['gallery'],
        bind(ROOT/'programs/rc_roma_m_inside_v1.py'),
        bind(ROOT/'programs/run_rc_h593_subset41_frozen_paths_v1.py'),
        bind(ROOT.parents[2]/'third_party/RoMaV2/src/romav2/matcher.py')]
    records = []; maximum_margin_error = 0.; maximum_native_mass_error = 0.; logits_checked = 0
    for r in opened['rows']:
        directory = SUBSET/f"query{r['index']:03d}"
        validation = read(directory/'validation.json')
        assert validation['status'] == 'SUBSET41_COST1_FIXED_PATH_CPU_PASS'
        payload = read(checked(validation['payload']))
        assert payload['row']['query_id'] == r['query_id']
        checked(payload['features'])
        inside_dir = INSIDE/f"query{r['index']:03d}"
        iv = read(inside_dir/'inside_validation.json'); assert iv['status']=='M_INSIDE_PATHS_PASS'
        inside = read(checked(iv['payload']))
        nv = read(inside_dir/'validation.json'); native_inside = read(checked(nv['payload']))
        axis = payload['row']['candidate_physical_rows']; winner = payload['row']['winner']
        challengers = payload['row']['challenger_positions']
        assert axis == [x['physical_row'] for x in inside['records']] == native_inside['candidate_physical_rows']
        target_positions = [i for i,g in enumerate(axis) if gallery[g] == old[r['query_id']]['identity']]
        assert len(target_positions) <= 1
        target = target_positions[0] if target_positions else None
        zs = {}
        for name, saved in payload['models'].items():
            z = [0.]*128
            for i,h in zip(challengers,saved['logits_hex']):
                z[i] = float.fromhex(h)
            selected = max(challengers,key=lambda i:z[i]); selected = selected if z[selected]>0 else winner
            assert axis[selected] == saved['selected'] == r['models'][name]['selected']
            assert (gallery[axis[selected]]==old[r['query_id']]['identity']) == r['models'][name]['correct']
            if target is not None:
                wrong = max((i for i in range(128) if i!=target),key=lambda i:z[i])
                err = abs(z[target]-z[wrong]-r['models'][name]['target_margin'])
                maximum_margin_error = max(maximum_margin_error,err)
                assert err < 1e-12
            zs[name] = z; logits_checked += len(challengers)
        native = zs['FULL_UV/native/HR1']
        original_wrong = max((i for i in range(128) if i!=target),key=lambda i:native[i]) if target is not None else None
        original_mass = [x['branches']['A1J1P1']['HR1'] for x in inside['records']]
        mass_wrong = max((i for i in range(128) if i!=target),key=lambda i:original_mass[i]) if target is not None else None
        record = dict(query_id=r['query_id'], original_query_id=r['original_query_id'], index=r['index'],
            fold=r['fold'], component=r['component'], target_in_C128=target is not None,
            raw_correct=r['raw_correct'], native_correct=r['native_correct'],
            original_rescue=r['native_correct'] and not r['raw_correct'],
            target_physical_row=axis[target] if target is not None else None,
            original_wrong_head_physical_row=axis[original_wrong] if original_wrong is not None else None,
            original_wrong_M_physical_row=axis[mass_wrong] if mass_wrong is not None else None,
            factorial={}, mass_factorial={}, visual={}, source_validation=bind(directory/'validation.json'),
            source_payload=validation['payload'], inside_validation=bind(inside_dir/'inside_validation.json'),
            inside_payload=iv['payload'])
        for stage in STAGES:
            mass_worlds = {}
            for factor in FACTORS:
                m = [x['branches'][factor][stage] for x in inside['records']]
                assert min(m)>0
                if factor=='A1J1P1':
                    maximum_native_mass_error = max(maximum_native_mass_error,
                        max(abs(x-y)for x,y in zip(m,native_inside['masses']['NATIVE'][stage])))
                new_wrong = max((i for i in range(128)if i!=target),key=lambda i:m[i]) if target is not None else None
                mass_worlds[factor] = dict(candidate_mean_M=avg(m), candidate_mean_logM=avg(list(map(math.log,m))),
                    target_M=m[target] if target is not None else None,
                    target_vs_original_M_wrong_logratio=math.log(m[target]/m[mass_wrong]) if target is not None else None,
                    target_vs_original_head_wrong_logratio=math.log(m[target]/m[original_wrong]) if target is not None else None,
                    target_vs_new_M_wrong_logratio=math.log(m[target]/m[new_wrong]) if target is not None else None,
                    new_max_M_wrong=axis[new_wrong] if new_wrong is not None else None)
            fields = ('candidate_mean_M','candidate_mean_logM','target_vs_original_M_wrong_logratio',
                      'target_vs_original_head_wrong_logratio','target_vs_new_M_wrong_logratio')
            record['mass_factorial'][stage] = dict(worlds=mass_worlds,
                contrasts={field:contrasts({k:v[field]for k,v in mass_worlds.items()})for field in fields})
            for mode in MODES:
                worlds = {}
                for factor in FACTORS:
                    path = head_path(mode,factor,stage); z = zs[path]
                    new_wrong = max((i for i in range(128)if i!=target),key=lambda i:z[i]) if target is not None else None
                    joined = r['models'][path]
                    worlds[factor] = dict(selected=joined['selected'], correct=joined['correct'], switched=joined['switched'],
                        target_logit=z[target] if target is not None else None,
                        target_vs_fixed_original_wrong=z[target]-z[original_wrong] if target is not None else None,
                        target_vs_new_wrong=z[target]-z[new_wrong] if target is not None else None,
                        wrong_reselection_term=z[original_wrong]-z[new_wrong] if target is not None else None,
                        strongest_wrong=axis[new_wrong] if target is not None else None)
                record['factorial'][mode+'/'+stage] = dict(worlds=worlds,
                    contrasts={field:contrasts({k:v[field]for k,v in worlds.items()})for field in
                               ('target_logit','target_vs_fixed_original_wrong','target_vs_new_wrong','wrong_reselection_term')})
        for arm in VISUAL_ARMS:
            record['visual'][arm] = {}
            for mode in MODES:
                path=mode+'/visual/'+arm+'/HR1';z=zs[path];nr=r['models'][path]
                new_wrong=max((i for i in range(128)if i!=target),key=lambda i:z[i]) if target is not None else None
                record['visual'][arm][mode]=dict(selected=nr['selected'],correct=nr['correct'],switched=nr['switched'],
                    target_logit=z[target] if target is not None else None,
                    fixed_original_wrong_margin_change=(z[target]-z[original_wrong])-(native[target]-native[original_wrong]) if target is not None else None,
                    dynamic_wrong_margin_change=(z[target]-z[new_wrong])-(native[target]-native[original_wrong]) if target is not None else None,
                    original_rescue_retained=bool(record['original_rescue']and nr['correct']))
        record['component_diagnostics']=component_diagnostics(inside,target,original_wrong,mass_wrong)
        records.append(record)
        sources += [bind(directory/'validation.json'),validation['payload'],payload['features'],
                    bind(inside_dir/'inside_validation.json'),iv['payload'],bind(inside_dir/'validation.json'),nv['payload']]
        sources += inside['sources']
    assert len(records)==41 and maximum_margin_error<1e-12 and maximum_native_mass_error<2e-12, (len(records),maximum_margin_error,maximum_native_mass_error)
    target_records=[r for r in records if r['target_in_C128']]
    rescue_records=[r for r in records if r['original_rescue']]
    assert len(target_records)==38 and len(rescue_records)==3
    summary={}
    for mode in MODES:
        for stage in STAGES:
            key=mode+'/'+stage
            summary[key]={'worlds':{},'contrasts':{}}
            for factor in FACTORS:
                worlds=[r['factorial'][key]['worlds'][factor]for r in records]
                summary[key]['worlds'][factor]=dict(correct=sum(x['correct']for x in worlds),denominator=41,
                    original3_rescues_retained=sum(r['factorial'][key]['worlds'][factor]['correct']for r in rescue_records),
                    changed_vs_native=sum(x['selected']!=o['factorial'][mode+'/HR1']['worlds']['A1J1P1']['selected']for x,o in zip(worlds,records)))
            for field in ('target_logit','target_vs_fixed_original_wrong','target_vs_new_wrong','wrong_reselection_term'):
                names=target_records[0]['factorial'][key]['contrasts'][field]
                summary[key]['contrasts'][field]={name:grouped(target_records,[r['factorial'][key]['contrasts'][field][name]for r in target_records])for name in names}
    mass_summary={}
    for stage in STAGES:
        mass_summary[stage]={}
        for field in target_records[0]['mass_factorial'][stage]['contrasts']:
            rr=records if field.startswith('candidate_mean')else target_records
            names=rr[0]['mass_factorial'][stage]['contrasts'][field]
            mass_summary[stage][field]={name:grouped(rr,[r['mass_factorial'][stage]['contrasts'][field][name]for r in rr])for name in names}
        # Same target-present population as target/wrong contrast, for a direct
        # comparison of overall confidence shift and relative selectivity.
        field='candidate_mean_logM';names=target_records[0]['mass_factorial'][stage]['contrasts'][field]
        mass_summary[stage]['candidate_mean_logM_target_present_only']={name:grouped(target_records,
            [r['mass_factorial'][stage]['contrasts'][field][name]for r in target_records])for name in names}
    concentration_summary={}
    for field in target_records[0]['component_diagnostics']['metrics']:
        rr=target_records if field.startswith('target_')else records
        concentration_summary[field]=grouped(rr,[r['component_diagnostics']['metrics'][field]for r in rr])
    concentration_summary['rank_counts']=dict(target_present=38,
        target_C_top1=sum(r['component_diagnostics']['metrics']['target_C_rank_desc']==1 for r in target_records),
        target_M_top1=sum(r['component_diagnostics']['metrics']['target_M_rank_desc']==1 for r in target_records),
        target_C_gt_original_M_wrong=sum(r['component_diagnostics']['metrics']['target_C_logratio_vs_original_M_wrong']>0 for r in target_records),
        target_C_gt_original_head_wrong=sum(r['component_diagnostics']['metrics']['target_C_logratio_vs_original_head_wrong']>0 for r in target_records))
    tensor_readiness=dict(verified_pair_files=sum(len(r['component_diagnostics']['candidates'])for r in records),
        complete_AJP_pair_files=[s for r in records for s in r['component_diagnostics']['full_tensor_sources']],
        all_pairs_have_directional_RMS_and_spatial16_summaries=True,
        scope='Complete direct head tensors are retained only for query000 first pair. Other pairs cannot replay P permutation from these summaries.')
    visual_same41={}
    for arm in VISUAL_ARMS:
        visual_same41[arm]={}
        for mode in MODES:
            visual_same41[arm][mode]=dict(correct=sum(r['visual'][arm][mode]['correct']for r in records),denominator=41,
                original3_rescues_retained=sum(r['visual'][arm][mode]['correct']for r in rescue_records),
                fixed_original_wrong_margin_change=grouped(target_records,[r['visual'][arm][mode]['fixed_original_wrong_margin_change']for r in target_records]),
                dynamic_wrong_margin_change=grouped(target_records,[r['visual'][arm][mode]['dynamic_wrong_margin_change']for r in target_records]))
    visual_v=read(VISUAL55/'validation.json');visual55=read(checked(visual_v['result']))
    sources += [bind(VISUAL55/'validation.json'),visual_v['result']]
    # New paired comparison of existing visual effects, within the same55 panel.
    visual_comparisons={}
    for side in ('Q','R'):
        for left,right in (('SHUFFLE','GRAY'),('LOWPASS','GRAY'),('SHUFFLE','LOWPASS')):
            key=side+'_'+left+'_minus_'+right
            vr=visual55['rows'];rec=[]
            for r in vr:
                ll=r['arms'][side+'_'+left];rr=r['arms'][side+'_'+right]
                rec.append(dict(query_id=r['query_id'],component=r['component'],
                    quality_dynamic_logratio_difference=ll['total_log_ratio_change']-rr['total_log_ratio_change'],
                    quality_fixed_logratio_difference=ll['target_vs_fixed_wrong_change']-rr['target_vs_fixed_wrong_change'],
                    M_ONLY_margin_difference=ll['heads']['M_ONLY']['total_margin_change']-rr['heads']['M_ONLY']['total_margin_change'],
                    FULL_UV_margin_difference=ll['heads']['FULL_UV']['total_margin_change']-rr['heads']['FULL_UV']['total_margin_change']))
            visual_comparisons[key]=dict(rows=rec,summary={f:grouped(rec,[r[f]for r in rec])for f in rec[0]if f not in ('query_id','component')})
    output=dict(status='EXISTING_AJP_FACTORIAL_FIXED_HEAD_LOCALIZATION_PASS',
        head='Original held-fold COST1 seven-parameter head; no new fit; natural C128; HOLD=0',
        panel=dict(all_queries=41,target_present=38,groups=len({r['component']for r in records}),
                   target_present_groups=len({r['component']for r in target_records}),native_correct=35,raw_correct=32,native_rescues=3),
        intervention='Direct A/J/P inputs to confidence predictor. Native geometry and refiner evidence held fixed. P is the expectation of Fourier sin/cos position features under softmatches derived from native J; J-off retains native P when P=1. P is not the Fourier embedding of the expected coordinate.',
        factorial_definition='P|A1J1=F111-F110; P|A1J0=F101-F100; JP|A1=F111-F110-F101+F100. Fixed original HR1 head competitor held across all eight worlds; dynamic rival analyzed separately.',
        scope=['Head margins are conditional computational effects in the original frozen model, not true image-level causal effects.',
               'Nonadditivity at final confidence/head output includes downstream sigmoid, aggregation and head feature nonlinearities.',
               'P-off removes the softmatch-weighted Fourier position-feature input; it does not remove all coordinate information from J/A or native warps.',
               'Visual shuffle also changes seams/context; lowpass changes several appearance cues. They are not equally strong or orthogonal interventions.',
               'Visual55 comparisons are separate52-target-present/31-group estimates; they are never pooled with the41-panel factorial.',
               'No ground-truth correspondence/object-localization annotations are used or validated.'],
        bootstrap=dict(unit='existing component group, resample group means with replacement',seed=BOOTSTRAP_SEED,
            draws=BOOTSTRAP_DRAWS,interval='95% percentile; descriptive opened-panel uncertainty; no multiple-comparison correction'),
        concentration_definition='For D=1024=2K sin/cos channels, directional mean ||P_i||^2/K = 2*rms(P)^2. Primary candidate C=(C_Q+C_R)/2; geometric mean reported as secondary. Observational ranks/correlations, no learned threshold.',
        factorial_summary=summary,mass_factorial_summary=mass_summary,visual_same41=visual_same41,
        concentration_summary=concentration_summary,tensor_readiness=tensor_readiness,
        visual55_paired_comparisons=visual_comparisons,source_files=sources,new_model_forwards=0,new_training=0)
    write(OUT/'rows41.json',records);write(OUT/'rescues3.json',rescue_records);write(OUT/'summary.json',output)
    (OUT/'report_zh.md').write_text(make_report(output,rescue_records))
    checks=dict(queries=41,head_logits_decoded_and_actions_recounted=logits_checked,
                maximum_margin_error=maximum_margin_error,maximum_native_mass_error=maximum_native_mass_error,
                original_native_correct=sum(r['native_correct']for r in records),all128_candidate_axes_matched=True,
                all_source_bindings_verified=True)
    write(OUT/'validation.json',dict(status=output['status'],checks=checks,script=bind(__file__),
        artifacts=[bind(OUT/name)for name in ('rows41.json','rescues3.json','summary.json','report_zh.md')],sources=sources))
    print(json.dumps(dict(status=output['status'],panel=output['panel'],checks=checks,
        factorial_M_ONLY_HR1=summary['M_ONLY/HR1'],factorial_FULL_UV_HR1=summary['FULL_UV/HR1'])))


if __name__=='__main__':
    main()
