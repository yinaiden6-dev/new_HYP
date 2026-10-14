"""Create evidence reports after the complete F03 token analysis."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import cv2
import numpy as np

NAMES={'line_break':'换行与末尾词位置','color':'颜色差异','text_structure':'文字排版结构','small_text':'细小文字差异','unclassified':'尚未人工分类'}


def read(path):return json.loads(path.read_text())


def table(headers, rows):
    clean=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(clean(x) for x in row)+' |' for row in rows])


def fmt(value):return '不可评估' if value is None else f'{value:+.6f}'


def weight_diagnostic(a):
    di=2*float(a['image_z_contribution']);dt=2*float(a['bm25_z_contribution'])
    if di>0 and dt<0:
        threshold=-dt/(di-dt)
        return f'图像权重 w > {threshold:.4f}',threshold
    if di<0 and dt>0:
        threshold=dt/(dt-di)
        return f'图像权重 w < {threshold:.4f}',threshold
    if di<=0 and dt<=0:return '两个信号均不支持正确候选；凸组合权重不能使此分差为正',None
    return '两个信号均支持正确候选',None


def focal_lines(folder, summary, rows, ocr, box):
    """Use polygon membership, even when a small line loses the best-line assignment."""
    width=summary['query_raw_grid']['width'];height=summary['query_raw_grid']['height']
    if summary['query_exif_orientation']>=5:width,height=height,width
    size=np.array([ocr['width'],ocr['height']],dtype=np.float32)
    bound=np.array(box['bbox'])/np.tile(size,2)
    focal=[]
    for i,line in enumerate(ocr['lines']):
        text=line['text'];lower=text.lower()
        kind='details_line' if 'details' in lower else 'number_unit_candidate' if re.search(r'\d.*(?:mg\b|ml\b|tablets|capsules|lbs\b|%)',lower) else None
        if kind is None:continue
        poly=np.array(line['polygon'],dtype=np.float32)/size;center=poly.mean(0)
        if not ((center>=bound[:2]) & (center<bound[2:])).all():continue
        matched=[]
        for row in rows:
            if not row['q_x_display']:continue
            x,y=float(row['q_x_display']),float(row['q_y_display'])
            if not ((np.array([x,y])>=bound[:2]) & (np.array([x,y])<bound[2:])).all():continue
            dx,dy=.5/width,.5/height
            cell=np.float32([[x-dx,y-dy],[x+dx,y-dy],[x+dx,y+dy],[x-dx,y+dy]])
            overlap=cv2.intersectConvexConvex(cell,poly)[0]*width*height
            if overlap>=.05:matched.append(row)
        total=sum(float(row['delta']) for row in matched)
        focal.append(dict(line=i,kind=kind,text=text,n=len(matched),delta_sum=total,
            contribution_to_full_mean_gap=total/summary['valid_tokens'],
            token_indices=json.dumps([int(row['q_token_index']) for row in matched]),
            ambiguity='Coarse patch overlaps this OCR line and may contain neighboring lines; regions can overlap'))
    with (folder/'focal_lines.csv').open('w',newline='') as f:
        fields=['line','kind','text','n','delta_sum','contribution_to_full_mean_gap','token_indices','ambiguity']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(focal)
    return focal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/token_mechanism_analysis/F03')
    out=parser.parse_args().output;data=read(out/'summary.json')
    inventory=list(csv.DictReader((out/'F03_error_inventory.csv').open()))
    inv={r['photo_id']:r for r in inventory}
    root=Path(__file__).resolve().parents[1]
    image_predictions={r['photo_id']:r for r in [json.loads(line) for line in
        (root/'results/colnomic7b/image/predictions.jsonl').read_text().splitlines()]}
    query_ocr=read(root/'cache/ocr/photos.json')
    boxes=read(root/'dataset/annotations/subject_boxes.json')['records']
    cv2.setNumThreads(2)
    contrasts={(r['target'],r['photo_id']):r for r in data['cases']}
    assert len(data['targets'])==len(inventory)==145
    links=[];paired=[];targets=[];weight_rows=[];diagnoses=[]
    for t in data['targets']:
        id=t['photo_id'];s=contrasts[id,id];targets.append(s);a=inv[id];folder=out/id/id
        rows=list(csv.DictReader((folder/'tokens.csv').open()))
        assert len(rows)==s['valid_tokens']
        assert abs(sum(float(r['contribution_to_mean_gap']) for r in rows)-s['image_gap'])<2e-6
        focal=focal_lines(folder,s,rows,query_ocr[id],boxes[id])
        stage='图像 Top-1 原本正确，融合后才变错。' if a['image_top1_correct']=='True' else '图像评分中已失败；编码器与 MaxSim 的责任尚未区分。'
        winner=image_predictions[id]['selected_reference_id']
        if winner!=s['negative']:
            stage+=f' 纯图像 Top-1 为 `{winner}`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。'
        identity='已有身份歧义标记，不能直接当作模型失败证据。' if t['identity_ambiguity'] else '按当前真值分析，未新增身份复核。'
        condition,threshold=weight_diagnostic(a)
        weight_rows.append(dict(photo_id=id,condition=condition,threshold=threshold,identity_ambiguity=t['identity_ambiguity']))
        worst=min(s['groups'],key=lambda g:g['contribution_to_mean_gap'])
        interpretation=(f'对完整图像分差最不利的区域是 `{worst["region"]}`，贡献 {fmt(worst["contribution_to_mean_gap"])}。'
            f'主体框内 token 占有效序列 {s["subject_tokens"]/s["valid_tokens"]:.1%}。'
            + ('固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。' if s['image_gap']<0<s['subject_only_gap'] else
               '主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。' if s['subject_only_gap']<0 else
               '主体 token 支持正确候选，仍需检查其他图库候选与最终融合。'))
        group=('身份歧义待复核' if t['identity_ambiguity'] else
               '融合覆盖正确图像赢家' if a['image_top1_correct']=='True' else
               '主体筛选可翻转本对候选' if s['image_gap']<0<s['subject_only_gap'] else
               '主体匹配仍偏错误候选' if s['image_gap']<0 else
               '纯图像另有错误赢家，当前负候选来自融合')
        diagnoses.append(dict(photo_id=id,diagnostic_group=group,image_top1_reference=winner,
                              F03_negative=s['negative'],same_image_and_fusion_winner=winner==s['negative'],
                              image_gap=s['image_gap'],subject_only_gap=s['subject_only_gap'],control=t['control']))
        parts=[f'# {id}\n',f'{identity} 用户指定主题：{NAMES[t["case_type"]]}。\n',
               f'正确参考 `{s["positive"]}`，F03 错误参考 `{s["negative"]}`。{stage}\n',
               '## 评分阶段\n',table(['检查','结果'],[
                   ['图像真值排名',a['image_rank']],['图像 Top-1 参考',winner],['BM25 真值排名',a['bm25_rank']],['融合真值排名',a['fusion_rank']],
                   ['图像正负分差',fmt(s['image_gap'])],['BM25 正负分差',fmt(s['bm25_gap'])],
                   ['0.5 × 图像 z 分差',fmt(float(a['image_z_contribution']))],
                   ['0.5 × BM25 z 分差',fmt(float(a['bm25_z_contribution']))],['最终融合分差',fmt(s['fusion_gap'])]]),
               '\n分差为正确参考减错误参考；归一化使用完整 5,405 行图库。\n',
               f'固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**{condition}**。'
               '此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。\n',
               '## Token 贡献\n',f'有效 token {s["valid_tokens"]} 个，主体框内 {s["subject_tokens"]} 个。'
               f'正向贡献 {fmt(s["positive_evidence"])}，负向贡献 {fmt(s["negative_evidence"])}，'
               f'相加等于图像分差 {fmt(s["image_gap"])}。缓存复现误差 {s["replay_max_error"]:.2g}。\n',
               table(['区域','Token 数','对完整均值分差的贡献','组内分差均值'],[
                   [g['region'],g['n'],fmt(g['contribution_to_mean_gap']),fmt(g['within_region_mean'])] for g in s['groups']]),
               f'\n仅保留主体 query token 的两候选分差：{fmt(s["subject_only_gap"])}。'
               '这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。\n',
               interpretation+'\n',
               '## 对应与独立几何\n',table(['候选','SIFT 内点','主体覆盖率','对齐通过阈值','残差中位数（网格单元）','超过 3 单元比例'],[
                   [name,g['inliers'],f'{g["coverage"]:.1%}',g['accepted'],fmt(s[key+'_geometry_distance_median_patches']),
                    '不可评估' if s[key+'_geometry_over_3_patches_fraction'] is None else f'{s[key+"_geometry_over_3_patches_fraction"]:.1%}']
                   for name,key,g in [('正确','positive',s['geometry'][0]),('错误','negative',s['geometry'][1])]]),
               f'\n主体中 {s["foreground_OCR_tokens"]} 个 token 与 OCR 相交，'
               f'{s["multiple_OCR_line_subject_tokens"]} 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。'
               '对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。\n']
        for title,key in [('最不利的 8 个 token',lambda r:float(r['delta'])),('最有利的 8 个 token',lambda r:-float(r['delta']))]:
            parts.extend(['\n## '+title+'\n',table(['索引','区域','Δi','Query OCR','正确参考 OCR','错误参考 OCR'],[
                [r['q_token_index'],r['region'],fmt(float(r['delta'])),r['q_ocr_text'],r['pos_ocr_text'],r['neg_ocr_text']]
                for r in sorted(rows,key=key)[:8]])])
        if focal:
            parts.extend(['\n## 末尾词／数字单位行的区域检查\n',
                '按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。'
                '这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。\n',
                table(['OCR 行','区域 token 数','对完整均值分差的区域贡献'],[
                    [f['text'],f['n'],fmt(f['contribution_to_full_mean_gap'])] for f in focal]),
                '\n[完整焦点行与 token 索引](focal_lines.csv)\n'])
        if t['control']:
            c=contrasts[id,t['control']];assert c['image_rank']==c['fusion_rank']==1;paired.append((s,c))
            parts.extend(['\n## 成功对照\n',f'`{c["photo_id"]}` 为同包装、图像和融合均正确的照片。'
                '按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。\n',
                table(['诊断','失败照片','成功照片'],[['图像分差',fmt(s['image_gap']),fmt(c['image_gap'])],
                    ['仅主体分差',fmt(s['subject_only_gap']),fmt(c['subject_only_gap'])],['主体 token 数',s['subject_tokens'],c['subject_tokens']]]),
                f'\n[对照 token 数据](../{c["photo_id"]}/tokens.csv) · [对照摘要](../{c["photo_id"]}/summary.json)\n'])
        else:parts.extend(['\n## 成功对照\n','本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。\n'])
        parts.extend(['\n## 图与原始数据\n','![对应连线](correspondence.png)\n','![分差贡献](contribution.png)\n',
            '[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)\n',
            '## 解释边界\n','本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。'
            '没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。\n'])
        if (folder/'focal_text.png').exists():parts.append('![末尾文字局部](focal_text.png)\n')
        (folder/'REPORT.md').write_text('\n'.join(parts))
        links.append([f'[{id}]({id}/{id}/REPORT.md)',NAMES[t['case_type']],a['image_rank'],a['fusion_rank'],
                      fmt(s['image_gap']),fmt(s['subject_only_gap']),t['control'] or '无',
                      '身份歧义' if t['identity_ambiguity'] else '待复核'])
    nonamb=[r for r in inventory if r['identity_ambiguity']=='False']
    negatives=sum(s['image_gap']<0 for s in targets)
    flips=sum(s['image_gap']<0 and s['subject_only_gap']>0 for s in targets)
    harms=sum(c['subject_only_gap']<0 for _,c in paired)
    both=sum(all(g['accepted'] for g in s['geometry']) for s in targets)
    stats=dict(targets=len(targets),paired_controls=len(paired),unique_control_photos=len({c['photo_id'] for _,c in paired}),
        missing_controls=len(targets)-len(paired),both_reference_geometry_accepted=both,
        subject_pair_flips_to_positive=flips,image_negative_pair_count=negatives,
        subject_pair_flips_to_negative_on_success_controls=harms,validated_score_conservation=True,
        max_score_replay_error=max(s['replay_max_error'] for s in data['cases']))
    (out/'verification.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    with (out/'fusion_weight_diagnostics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(weight_rows[0]));writer.writeheader();writer.writerows(weight_rows)
    with (out/'diagnostics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(diagnoses[0]));writer.writeheader();writer.writerows(diagnoses)
    highlights=[]
    for id in NAMES_CASES:
        s=contrasts[id,id];a=inv[id]
        highlights.append([f'[{NAMES[s["case_type"]]}]({id}/{id}/REPORT.md)',a['image_rank'],a['fusion_rank'],fmt(s['image_gap']),
            fmt(float(a['image_z_contribution'])),fmt(float(a['bm25_z_contribution'])),fmt(s['subject_only_gap'])])
    report=['# F03 全部 145 个错误：Token 对应与贡献\n',
        '逐案复用冻结 embedding 与评分缓存，分析正确／错误候选的全部有效 token；没有修改 baseline。'
        '每例均附独立报告、完整对应 CSV、贡献分解、匹配连线和可用成功对照。\n',
        '## 已验证的失败阶段\n',table(['范围','图像已错','图像原本正确、融合变错','总计'],[
            ['全部 F03 错误',data['stage_counts']['image_already_wrong'],data['stage_counts']['fusion_overrode_correct_image_top1'],145],
            ['排除已有身份歧义',sum(r['failure_stage']=='image_already_wrong' for r in nonamb),
             sum(r['failure_stage']=='fusion_overrode_correct_image_top1' for r in nonamb),len(nonamb)]]),
        '\n19 例已有身份歧义标签，保留分析但不直接作为模型失败证据。'
        f'现有人工分类文件含 {data["manual_error_annotation_count"]} 条记录；四个主题来自用户，其余案例未编造语义类别。\n',
        '## 根据测量得到的逐案诊断分组\n',table(['诊断','案例数'],Counter(d['diagnostic_group'] for d in diagnoses).items()),
        '\n这些是评分与筛选行为的分组，不是已经验证的编码器因果机制。F03 的错误候选可能与纯图像 Top-1 不同；逐案报告分别记录两者。\n',
        '## 指定的四个案例\n',table(['案例','图像排名','融合排名','图像分差','0.5图像z差','0.5 BM25 z差','仅主体分差'],highlights),
        '\n颜色案例的图像结果正确，应首先追查融合。文字结构案例图像排名 148，融合升至第 10，仍失败。'
        '20/50 mL 和 100/250 mL 案例的图像分差为负。容量、NDC、其他细节也有变化，不能当作只改变排版的受控实验。\n',
        '## 覆盖与对照\n',table(['检查','结果'],[['完成案例','145/145'],['同包装成功对照',len(paired)],
            ['不同成功照片',stats['unique_control_photos']],['缺少对照',stats['missing_controls']],['正负几何均通过阈值',both],
            ['主体 token 筛选使两候选由负翻正',f'{flips}/{negatives}'],['同样筛选损害成功对照的两候选分差',f'{harms}/{len(paired)}'],
            ['最大缓存分数复现误差',stats['max_score_replay_error']]]),
        '\n对照可重复使用，不能视为独立样本。可见包装面没有全部人工确认。主体筛选翻转仅指这两个候选，'
        '不是完整图库 Recall@1 改善，也不是裁剪图片重新编码的效果。\n',
        '## 方法与结论边界\n',
        '- MaxSim 使用有效 query token 均值，保留有效非图像 token；去掉零向量，FP32 点积，不再次归一化。每例验证分差守恒。\n'
        '- 网格从原始解码方向按 EXIF 转至显示坐标，再关联主体框与 OCR；patch 面积交叠至少 5% 才分配行，同时记录多行歧义。\n'
        '- 独立 SIFT 对齐：query 主体框、完整参考图、比率 .75、最大边 1400、RANSAC 4px、至少 8 内点且覆盖主体 15%。自动通过仍非人工几何真值。\n'
        '- 匹配图固定展示主体中最有利／最不利各 8 个 token；3 网格单元空间残差仅用于描述，没有被验证为异常阈值。\n'
        '- 贡献是差额核算。上下文特征、框内遮挡、展开图重复面和透视限制语义归因。编码信息可分性需要 feature probe，像素级因果作用需要受控干预。\n',
        '## 数据与复现\n','[错误评分阶段清单](F03_error_inventory.csv) · [全部摘要](cohort_summary.csv) · '
        '[融合权重双候选诊断](fusion_weight_diagnostics.csv) · [覆盖和数值验证](verification.json) · [完整 JSON 与缓存签名](summary.json)\n',
        '[逐案诊断分组与图像赢家](diagnostics.csv)\n',
        '在已装 ML 依赖的环境运行：\n\n```bash\npython -B scripts/analyze_colnomic_tokens.py --all-errors --resume\n'
        'python -B scripts/report_colnomic_tokens.py\n```\n',
        '## 全部逐案报告\n',table(['案例','用户主题','图像排名','融合排名','图像分差','仅主体分差','成功对照','身份状态'],links)]
    (out/'REPORT.md').write_text('\n'.join(report));print(json.dumps(stats,ensure_ascii=False))


NAMES_CASES=['photo_41de3eb383934ec3','photo_21bd7df9ed6795e7','photo_1f48d45127db4129','photo_0e87e0717ab1c215']
if __name__=='__main__':main()
